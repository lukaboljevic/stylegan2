from typing import Callable, TypeVar, cast


C = TypeVar("C", bound=Callable)


def proxy(f: C) -> C:
    """
    Proxy function signature map for `Module.__call__` type hint.
    """
    return cast(C, lambda self, *x, **y: super(self.__class__, self).__call__(*x, **y))
