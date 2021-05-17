from collections import defaultdict


class NestedDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))


def assure_instance_type(data, check_type=list):
    """Checks and corrects instance type"""
    if not isinstance(data, check_type):
        data = type(data)
    return data