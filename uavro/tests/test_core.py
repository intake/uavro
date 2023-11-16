import json
import os
import pytest
from uavro.core import read, dask_read_avro
from uavro.tests.util import pdata, ldata, t

here = os.path.dirname(__file__)
fn = os.path.join(here, 'twitter.avro')
expected = [json.loads(l) for l in open(os.path.join(here, 'twitter.json'))]


def test_basic():
    # https://github.com/miguno/avro-cli-examples/
    out = read(fn)
    assert out.to_dict(orient='records') == expected


def test_official():
    # official test data
    # https://github.com/apache/avro/blob/master/share/test/data/
    pytest.importorskip('snappy')
    out = read(os.path.join(here, 'weather.avro'))
    outs = read(os.path.join(here, 'weather-snappy.avro'))
    expected = [json.loads(l) for l in open(os.path.join(here, 'weather.json'))]
    assert out.to_dict(orient='records') == expected
    assert outs.to_dict(orient='records') == expected


def test_primitive():
    out = read(os.path.join(here, 'primitive.avro'))
    assert len(out) == 1000
    assert len(out.columns) == 8
    for k in pdata[0]:
        if k == 'e':
            # 32-bit float loses precision
            assert abs(out[k][0] - pdata[0][k]) < 0.0001
        else:
            assert out[k][0] == pdata[0][k]


def test_logical():
    out = read(os.path.join(here, 'logical.avro'))
    assert len(out) == 1000
    assert len(out.columns) == 4
    assert out.iloc[0, 0].microsecond % 1000 == 0  # ms resolution
    assert out.iloc[0, 1].microsecond == t.microsecond  # us resolution
    assert (abs(out.f - 0.123) < 0.0001).all()


def test_with_dask_chunk():
    pytest.importorskip('dask.dataframe')
    N = 100
    df = dask_read_avro([fn] * N)
    out = df.compute()
    assert out.to_dict(orient='records') == expected * N
    df = dask_read_avro([fn] * N, blocksize=1000)
    assert df.npartitions > 2
    out = df.compute()
    assert out.to_dict(orient='records') == expected * N


def test_with_dask_file():
    pytest.importorskip('dask.dataframe')
    N = 100
    df = dask_read_avro([fn] * N, blocksize=None)
    out = df.compute()
    assert out.to_dict(orient='records') == expected * N
