import numpy as np
cimport numpy as np
from libc.stdint cimport (
    int8_t, int16_t, int32_t, int64_t,
    uint8_t, uint16_t, uint32_t, uint64_t
)
from libc.string cimport memcpy


cdef class bytesIO:
    cdef char * buff
    cdef int loc
    cdef int size

    def __init__(self, bytes data):
        self.size = len(data)
        self.buff = data
        self.loc = 0

    cdef int check(self, int n):
        return self.loc + n < self.size

    cdef void advance(self, int n):
        self.loc += n

    cdef char * data(self):
        return self.buff + self.loc

    def read(self, int i):
        if self.loc >= self.size:
            return b''
        self.loc += i
        if self.loc > self.size:
            return self.buff[self.loc - i:self.size]
        return self.buff[self.loc - i:self.loc]

    cpdef seek(self, int i):
        self.loc = i

    cpdef tell(self):
        return self.loc


cdef int64_t read_int(bytesIO data):
    cdef uint64_t result = 0
    cdef short shift = 0
    cdef uint8_t byte

    while True:
        byte = data.data()[0]
        data.advance(1)
        result |= <long>(byte & 0x7f) << shift
        shift += 7
        if byte >> 7 == 0:
            break
    return (result >> 1) ^ -(result & 1)


cdef double read_double(bytesIO data):
    cdef double out
    memcpy(&out, data.data(), 8)
    data.advance(8)
    return out


cdef float read_float(bytesIO data):
    cdef float out
    memcpy(&out, data.data(), 4)
    data.advance(4)
    return out


cdef char read_bool(bytesIO data):
    cdef char out
    out = data.data()[0] > 0
    data.advance(1)
    return out


cdef bytes read_bytes(bytesIO data):
    cdef int64_t size
    cdef bytes out
    size = read_int(data)
    out = data.data()[:size]
    data.advance(size)
    return out


def read(arrs, data, schema, int nrows, int off=0):
    cdef int ncols
    cdef list types, arr
    cdef bytesIO f
    ncols = len(arrs)
    types = [s['type'] for s in schema]
    arr = [arrs[s['name']] for s in schema]
    f = bytesIO(data)
    for i in range(off, nrows + off):
        for j in range(ncols):
            t = types[j]
            if t == 'long':
                arr[j][i] = read_int(f)
            elif t == 'int':
                arr[j][i] = read_int(f)
            elif t == 'double':
                arr[j][i] = read_double(f)
            elif t == 'string':
                arr[j][i] = read_bytes(f).decode('utf8')
            elif t == 'bytes':
                arr[j][i] = read_bytes(f)
            elif t == 'bool':
                arr[j][i] = read_bool(f)
            else:
                raise ValueError()
