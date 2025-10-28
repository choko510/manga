"""Type definitions used by the Hitomi.la Python port."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, MutableSet
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union


JsonObject = Dict[str, "JsonValue"]
JsonValue = Union[str, int, float, bool, None, "JsonObject", List["JsonValue"]]


class IdSet(MutableSet[int]):
    """Order-preserving set that mirrors JavaScript's ``Set`` behaviour."""

    def __init__(self, iterable: Optional[Iterable[int]] = None, *, is_negative: bool = False) -> None:
        self._items: "OrderedDict[int, None]" = OrderedDict()
        self.is_negative = is_negative
        if iterable is not None:
            for value in iterable:
                self.add(value)

    def __contains__(self, value: object) -> bool:  # type: ignore[override]
        try:
            key = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return key in self._items

    def __iter__(self) -> Iterator[int]:  # type: ignore[override]
        return iter(self._items.keys())

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._items)

    def add(self, value: int) -> None:  # type: ignore[override]
        self._items[int(value)] = None

    def discard(self, value: int) -> None:  # type: ignore[override]
        self._items.pop(int(value), None)

    def to_list(self) -> List[int]:
        return list(self._items.keys())


Node = Tuple[List[bytes], List[Tuple[int, int]], List[int]]


@dataclass(slots=True)
class Title:
    display: str
    japanese: Optional[str]


@dataclass(slots=True)
class LanguageName:
    english: Optional[str]
    local: Optional[str]


TagType = Union[
    "artist",
    "group",
    "type",
    "language",
    "series",
    "character",
    "male",
    "female",
    "tag",
]


@dataclass(slots=True)
class Tag:
    type: str
    name: str
    is_negative: bool = False


@dataclass(slots=True)
class Image:
    index: int
    hash: str
    name: str
    has_avif: bool
    has_webp: bool
    has_jxl: bool
    width: int
    height: int


@dataclass(slots=True)
class GalleryTranslation:
    id: int
    language_name: LanguageName


GalleryType = Union["doujinshi", "manga", "artistcg", "gamecg", "anime"]


@dataclass(slots=True)
class Gallery:
    id: int
    title: Title
    type: GalleryType
    language_name: LanguageName
    artists: List[str]
    groups: List[str]
    series: List[str]
    characters: List[str]
    tags: List[Tag]
    files: List[Image]
    published_date: datetime | None
    translations: List[GalleryTranslation]
    related_ids: List[int]


PopularityPeriod = Union["day", "week", "month", "year"]


StartingCharacter = Union[
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "0-9",
]
