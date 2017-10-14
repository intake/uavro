import datetime
import decimal

pdata = [{'a': 'hello©', 'b': b'hello', 'c': 123, 'd': 786120,
         'e': 0.123, 'f': 0.123, 'g': b'0000', 'h': 'DIAMONDS'}]


def generate_test_data_primitive(fn):
    import fastavro
    schema = {"namespace": "example.avro",
              "type": "record",
                      "name": "User",
              "fields": [
                  {"name": "a", "type": "string"},
                  {"name": "b", "type": "bytes"},
                  {"name": "c", "type": "int"},
                  {"name": "d", "type": "long"},
                  {"name": "e", "type": "float"},
                  {"name": "f", "type": "double"},
                  {"name": "g", "type": {'type': "fixed", 'size': 4}},
                  {"name": "h", "type": {'type': "enum",
                   "symbols": ["SPADES", "HEARTS", "DIAMONDS", "CLUBS"]}},
                  ]
              }
    with open(fn, 'wb') as out:
        fastavro.writer(out, schema, pdata * 1000)


t = datetime.datetime(2017, 10, 14, 17, 34, 10, 282667)
d = datetime.datetime(2017, 10, 14)
dd = d.date()
dt = t - d
ldata = [{'a': t, 'b': t, 'c': dt, 'd': dt, 'e': dd, 'f': 0.123, 'g': 0.123}]
ldata = [{'a': t, 'b': t, 'e': dd, 'f': decimal.Decimal('0.123')}]


def generate_test_data_logical(fn):
    import fastavro
    # fastavro does not support time-delta or decimal-in-fixed
    schema = {"namespace": "example.avro",
              "type": "record",
                      "name": "User",
              "fields": [
                  {"name": "a", "type": {'type': "long",
                      'logicalType': 'timestamp-millis'}},
                  {"name": "b", "type": {'type': "long",
                      'logicalType': 'timestamp-micros'}},
                  {"name": "e", "type": {'type': "int",
                      'logicalType': 'date'}},
                  {"name": "f", "type": {'type': "bytes",
                      'logicalType': 'decimal', 'scale': 3}},
                  ]
              }
    with open(fn, 'wb') as out:
        fastavro.writer(out, schema, ldata * 1000)
