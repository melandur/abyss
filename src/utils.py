from collections import defaultdict


class NestedDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))


def check_instance(data, check_type=list):
    if not isinstance(data, check_type):
        data = type(data)
    return data
