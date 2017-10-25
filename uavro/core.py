from collections import OrderedDict
from fastparquet.dataframe import empty
import io
import json
import numpy as np
import os
try:
    import snappy
except ImportError:
    snappy = None
import zlib

from . import reader

MAGIC = b'Obj\x01'
SYNC_SIZE = 16


def read_long(fo):
    """variable-length, zig-zag encoding."""
    c = fo.read(1)
    b = ord(c)
    n = b & 0x7F
    shift = 7
    while (b & 0x80) != 0:
        b = ord(fo.read(1))
        n |= (b & 0x7F) << shift
        shift += 7
    return (n >> 1) ^ -(n & 1)


def read_bytes(fo):
    """a long followed by that many bytes of data."""
    size = read_long(fo)
    return fo.read(size)


typemap = {
    'boolean': np.dtype(bool),
    'int': np.dtype('int32'),
    'long': np.dtype('int64'),
    'float': np.dtype('float32'),
    'double': np.dtype('float64'),
    'enum': 'category'
}


def map_types(header, schema):
    types = OrderedDict()
    for entry in schema['fields']:
        # should bother with root record's name and namespace?
        if isinstance(entry['type'], dict):
            entry.update(entry['type'])
        if entry['type'] in ['record', 'array', 'map']:
            # TODO: as with fastparquet, would not be too bad to implement
            # one-level map/array or non-repetead, flattenable records.
            raise ValueError('uavro cannot read schemas containing '
                             'nested/repeated data types.')
        if 'logicalType' in entry:
            lt = entry['logicalType']
            if lt == 'decimal':
                t = np.dtype('float64')
            elif lt.startswith('time-'):
                t = np.dtype('timedelta64')
            elif lt.startswith('timestamp-') or lt == 'date':
                t = np.dtype('datetime64')
            elif lt == 'duration':
                t = np.dtype('S12')  # don't bother converting
        elif entry['type'] == 'fixed':
            t = np.dtype("S%s" % entry['size'])
        else:
            t = typemap.get(entry['type'], np.dtype("O"))
        types[entry['name']] = t
    header['dtypes'] = types


def read_header(fo):
    """Extract an avro file's header

    fo: file-like
        This should be in bytes mode, e.g., io.BytesIO

    Returns dict representing the header
    """
    assert fo.read(len(MAGIC)) == MAGIC, 'Magic avro bytes missing'
    meta = {}
    out = {'meta': meta}
    while True:
        n_keys = read_long(fo)
        if n_keys == 0:
            break
        for i in range(n_keys):
            key = read_bytes(fo).decode('utf8')
            val = read_bytes(fo)
            if key == 'avro.schema':
                val = json.loads(val.decode('utf8'))
                map_types(out, val)
                out['schema'] = val
            meta[key] = val
    out['sync'] = fo.read(SYNC_SIZE)
    out['header_size'] = fo.tell()
    fo.seek(0)
    out['head_bytes'] = fo.read(out['header_size'])
    peek_first_block(fo, out)
    return out


def peek_first_block(fo, out):
    fo.seek(out['header_size'])
    out['first_block_count'] = read_long(fo)
    out['first_block_bytes'] = read_long(fo)
    out['first_block_data'] = fo.tell()
    out['blocks'] = [{'offset': out['header_size'], 'doffset': fo.tell(),
                      'size': out['first_block_bytes'],
                      'nrows': out['first_block_count']}]


def scan_blocks(fo, header, file_size):
    """Find offsets of the blocks by skipping each block's data.

    Useful where the blocks are large compared to read buffers.
    If blocks are small compared to read buffers, better off searching for the
    sync delimiter.

    Results are attached to the header dict.
    """
    if len(header['blocks']) > 1:
        # already done
        return
    if len(header['blocks']) == 0:
        peek_first_block(fo, header)
    data = header['first_block_data']
    bytes = header['first_block_bytes']
    nrows = header['first_block_count']
    while True:
        off0 = data + bytes
        if off0 + SYNC_SIZE >= file_size:
            break
        fo.seek(off0)
        assert fo.read(SYNC_SIZE) == header['sync'], "Sync failed"
        off = fo.tell()
        num = read_long(fo)
        bytes = read_long(fo)
        data = fo.tell()
        if num == 0 or bytes == 0:
            # can have zero-length blocks??
            continue
        header['blocks'].append({'offset': off, 'size': data - off + bytes,
                                 'nrows': num, 'doffset': data})
        nrows += num
    header['nrows'] = nrows


def read_block_bytes(data, block, head, arrs, off):
    codec = head['meta'].get('avro.codec', b'null')
    data = decompress[codec](data)
    reader.read(arrs, data, head['schema']['fields'], block['nrows'], off)


def read(fn):
    """Simple entry point: read a file, output dataframe"""
    f = open(fn, 'rb')
    file_size = os.path.getsize(fn)
    head = read_header(f)
    return filelike_to_dataframe(f, file_size, head)


def make_empty(head):
    """Pre-assign dataframe to put values into"""
    cats = {e['name']: e['symbols'] for e in head['schema']['fields']
            if e['type'] == 'enum'}
    df, arrs = empty(head['dtypes'].values(), head['nrows'],
                     cols=head['dtypes'], cats=cats)

    for entry in head['schema']['fields']:
        # temporary array for decimal
        if entry.get('logicalType', None) == 'decimal':
            if entry['type'] == 'fixed':
                arrs[entry['name']] = np.empty(head['nrows'],
                                               'S%s' % entry['size'])
            else:
                arrs[entry['name']] = np.empty(head['nrows'], "O")
    return df, arrs


def filelike_to_dataframe(f, size, head, scan=True):
    """Read bytes, make dataframe

    The intent is to be able to pass a real file, or any file-like object,
    including BytesIO.

    Parameters
    ----------
    f: file-like instance
    size: int
        Number of bytes to read, often the whole available
    head: dict
        Parsed header information relating to this data. This allows for
        reading from bytes blocks part-way through a file.
    scan: bool
        Whether a block scan is required; if False, head must already contain
        a list of blocks values and nrows, the total number of rows.
    """
    if scan:
        scan_blocks(f, head, size)

    df, arrs = make_empty(head)
    off = 0

    for block in head['blocks']:
        f.seek(block['doffset'])
        data = f.read(block['size'])
        arrs = {k: v for (k, v) in arrs.items() if not k.endswith('-catdef')}
        read_block_bytes(data, block, head, arrs, off)
        off += block['nrows']

    convert_types(head, arrs, df)
    return df


def convert_types(head, arrs, df):
    for entry in head['schema']['fields']:
        # logical conversions
        lt = entry.get('logicalType', '')
        if lt.endswith('millis'):
            a = arrs[entry['name']].view('int64')
            a *= 1000000
        elif lt.endswith('micros'):
            a = arrs[entry['name']].view('int64')
            a *= 1000
        elif lt == 'date':
            # https://avro.apache.org/docs/1.8.2/spec.html#Date
            # says days since 1970, but fastparquet uses fromordinal
            a = arrs[entry['name']].view('int64')
            a *= 1000000000 * 24 * 3600
        elif lt == 'decimal':
            # https://avro.apache.org/docs/1.8.2/spec.html#Decimal
            # fails on py2
            scale = 10**-entry['scale']
            df[entry['name']].values[:] = [int.from_bytes(b, 'big') * scale
                                           for b in arrs[entry['name']]]



decompress = {b'snappy': lambda d: snappy.decompress(d[:-4]),
              b'deflate': lambda d: zlib.decompress(d, -15),
              b'null': lambda d: d}


def dask_read_avro(urlpath, block_finder='auto', blocksize=100000000,
                   **kwargs):
    """Read set of avro files into dask dataframes

    Use this only with avro schema that make sense as tabular data, i.e.,
    not deeply nested with arrays and maps.

    Parameters
    ----------
    urlpath: string
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``), or globstring pointing to data.
    block_finder: auto|scan|seek|none
        Method for chunking avro files.
        - scan: read the first bytes of every block to find the size of all
            blocks and therefore the boundaries
        - seek: find the block delimiter bytestring every blocksize bytes
        - none: do not chunk files, parallelism will be only across files
        - auto: use seek if the first block is larger than 0.2*blocksize, else
            scan.
    blocksize: int
        maybe used by the block-finder (see above)
    """
    from dask import delayed
    from dask.bytes.core import get_fs_paths_myopen
    from dask.bytes.utils import seek_delimiter
    from dask.dataframe import from_delayed

    if block_finder not in ['auto', 'scan', 'seek', 'none']:
        raise ValueError("block_finder must be in ['auto', 'scan', 'seek',"
                         " 'none'], got %s" % block_finder)
    fs, paths, myopen = get_fs_paths_myopen(urlpath, None, 'rb', None, **kwargs)
    chunks = []
    dread = delayed(dask_read_chunk)
    head = None
    for path in paths:
        if head is None:
            # sample first file
            with myopen(path, 'rb') as f:
                head = read_header(f)
        size = fs.size(path)
        b_to_s = blocksize / head['first_block_bytes']
        if (block_finder == 'none' or blocksize > 0.9 * size or
                head['first_block_bytes'] > 0.9 * size):
            # one chunk per file
            chunks.append(dread(path, myopen, 0, size, head))
        elif block_finder == 'scan' or (block_finder == 'auto' and b_to_s < 0.2):
            # hop through file pick blocks ~blocksize apart, append to chunks
            with myopen(path, 'rb') as f:
                head['blocks'] = []
                scan_blocks(f, head, size)
            blocks = head['blocks']
            head['blocks'] = []
            loc0 = head['header_size']
            loc = loc0
            nrows = 0
            for o in blocks:
                loc += o['size'] + SYNC_SIZE
                nrows += o['nrows']
                if loc - loc0 > blocksize:
                    chunks.append(dread(path, myopen, loc0, loc - loc0, head,
                                       scan=False))
                    loc0 = loc
                    nrows = 0
            chunks.append(dread(path, myopen, loc0, size - loc0, head,
                                scan=False))
        else:
            # block-seek case: find sync markers
            loc0 = head['header_size']
            with myopen(path, 'rb') as f:
                while True:
                    f.seek(blocksize, 1)
                    seek_delimiter(f, head['sync'], head['first_block_bytes']*4)
                    loc = f.tell()
                    chunks.append(dread(path, myopen, loc0, loc - loc0, head))
                    if f.tell() >= size:
                        break
                    loc0 = loc
    return from_delayed(chunks, meta=head['dtypes'],
                        divisions=[None] * (len(chunks) + 1))


def dask_read_chunk(path, myopen, offset, size, head, scan=True):
    with myopen(path, 'rb') as f:
        f.seek(offset)
        data = f.read(size)
    return filelike_to_dataframe(io.BytesIO(data), size, head, scan=scan)
