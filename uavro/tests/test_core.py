import json
import os
from uavro.core import read

here = os.path.dirname(__file__)


def test_basic():
    out = read(os.path.join(here, 'twitter.avro'))
    expected = [json.loads(l) for l in open(os.path.join(here, 'twitter.json'))]
    assert out.to_dict(orient='records') == expected