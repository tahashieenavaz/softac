from collections.abc import ItemsView
from typing import Any


class SoftActorCritic:
    def __init__(self):
        self.__set_attributes(locals().items())

    def __set_attributes(self, items: ItemsView[str, Any]) -> None:
        for key, value in items:
            if key == "self":
                continue
            setattr(self, key, value)
