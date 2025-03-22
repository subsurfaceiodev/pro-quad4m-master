import numpy

INDENT = 3
SPACE = " "
NEWLINE = "\n"


def to_json(o, level=0):
    ret = ""
    if isinstance(o, dict):
        ret += "{" + NEWLINE
        comma = ""
        for k, v in o.items():
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level + 1)
            ret += '"' + str(k) + '":' + SPACE
            ret += to_json(v, level + 1)

        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        ret += '"' + o.replace('"', '\"') + '"'
    elif isinstance(o, list):
        ret += "[" + ", ".join([to_json(e, level + 1) for e in o]) + "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, float):
        if not o % 1:
            ret += '%.1f' % o
        else:
            ret += '%.7g' % o
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, numpy.ndarray) and numpy.issubdtype(o.dtype, numpy.integer):
        ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
    elif isinstance(o, numpy.ndarray) and numpy.issubdtype(o.dtype, numpy.inexact):
        ret += "[" + ','.join(['%.7g' % x for x in o.flatten().tolist()]) + "]"
    elif not o:
        ret += 'null'
    else:
        raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
    return ret
