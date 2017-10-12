from collections import OrderedDict
import json
from fastparquet.dataframe import empty
import numpy as np
import os
import snappy
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
    'double': np.dtype('float64')
}


def map_types(header, schema):
    types = OrderedDict()
    for entry in schema['fields']:
        # should bother with root record's name and namespace?
        if isinstance(entry['type'], dict):
            entry['type'] = entry['type']['type']
        if 'logicalType' in entry:
            lt = entry['logicalType']
            if lt == 'decimal':
                t = np.dtype('float64')
            elif lt.startswith('time-'):
                t = np.dtype('timedelta64')
            elif lt.startswith('timestamp-') or lt == 'date':
                t = np.dtype('datetime64')
            elif lt == 'duration':
                t = np.dtype("O")  # three-element tuples/arrays
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
    f = open(fn, 'rb')
    file_size = os.path.getsize(fn)
    head = read_header(f)
    scan_blocks(f, head, file_size)

    df, arrs = empty(head['dtypes'].values(), head['nrows'],
                     cols=head['dtypes'])
    off = 0

    for block in head['blocks']:
        f.seek(block['doffset'])
        data = f.read(block['size'])
        read_block_bytes(data, block, head, arrs, off)
        off += block['nrows']
    return df


decompress = {b'snappy': lambda d: snappy.decompress(d[:-4]),
              b'deflate': lambda d: zlib.decompress(d, -15),
              b'null': lambda d: d}

