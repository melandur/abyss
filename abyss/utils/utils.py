from collections import defaultdict
from typing import Any


class NestedDefaultDict(defaultdict):
    """Dynamically expandable nested dict"""

    def __init__(self, *args, **kwargs):
        super().__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))


def assure_instance_type(data, check_type=list) -> Any:
    """Checks and corrects instance type"""
    if not isinstance(data, check_type):
        raise ValueError(f'{data} is type: {type(data)}, expected: {check_type}')
    return data
