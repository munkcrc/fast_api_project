from typing import Any, List, Tuple


def _none_empty_attributes(object_, attributes: List[str]) -> List[Tuple[str, Any]]:
    return [
        (key, getattr(object_, key)) for key in attributes
        if getattr(object_, key) is not None and (
                    not hasattr(getattr(object_, key), '__len__') or len(
            getattr(object_, key)) > 0)]


def _repr(object_, attributes: List[str]) -> str:
    return (f'{object_.__class__.__name__}'
            f"({', '.join([f'{t[0]}={t[1]!r}' for t in _none_empty_attributes(object_, attributes)])})")


def _str(object_, attributes: List[str]) -> str:
    return (f'{object_.__class__.__name__}'
            f"({', '.join([f'{t[1]!s}' for t in _none_empty_attributes(object_, attributes)])})")


def _eq(object_self, object_other, attributes: List[str]):
    if object_other.__class__ is not object_self.__class__:
        return NotImplemented
    return [getattr(object_self, attribute) for attribute in attributes] == [
        getattr(object_other, attribute) for attribute in attributes]


