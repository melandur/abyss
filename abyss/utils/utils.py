from collections import defaultdict
from typing import Any


def assure_instance_type(data, check_type=list) -> Any:
    """Checks and corrects instance type"""
    if not isinstance(data, check_type):
        raise ValueError(f'{data} is type: {type(data)}, expected: {check_type}')
    return data


class NestedDefaultDict(defaultdict):
    """Nested dict, which can be dynamically expanded
    https://stackoverflow.com/questions/19189274/nested-defaultdict-of-defaultdict"""

    def __init__(self, *args, **kwargs):
        super().__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))
