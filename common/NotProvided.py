from typing import Any, Callable


class NotProvided:
    """Default values for constructors etc. Is already backed into Python? Well, this
    blows up when serialised with jsonloader."""

    @staticmethod
    def is_not_provided(obj) -> bool:
        if isinstance(obj, type):
            return obj == NotProvided
        return isinstance(obj, NotProvided)

    @staticmethod
    def is_provided(obj) -> bool:
        return not NotProvided.is_not_provided(obj)
        return not isinstance(obj, NotProvided)

    @staticmethod
    def value_if_not_provided(obj, value_if_not_provided: Any) -> Any:
        if NotProvided.is_not_provided(obj):
            return value_if_not_provided
        return obj

    @staticmethod
    def value_if_provided(obj, f_if_provided: Callable[[Any], Any]) -> Any:
        if NotProvided.is_provided(obj):
            return f_if_provided(obj)
        return obj

    @staticmethod
    def none_if_not_provided(obj: Any) -> Any:
        return NotProvided.value_if_not_provided(obj, None)

    def __init__(self):
        pass
