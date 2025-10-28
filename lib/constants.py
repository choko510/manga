"""Constants for the Hitomi.la API wrapper."""
from __future__ import annotations

from enum import Enum, auto
from typing import Final, Tuple


class ErrorCode(Enum):
    """Enumerates the custom error codes used by the library."""

    INVALID_VALUE = auto()
    INVALID_CALL = auto()
    DUPLICATED_ELEMENT = auto()
    LACK_OF_ELEMENT = auto()
    REQUEST_REJECTED = auto()


RAW_GALLERY_KEYS: Final[Tuple[str, ...]] = ("parody", "artist", "group", "character")

TAG_TYPES: Final[frozenset[str]] = frozenset(
    RAW_GALLERY_KEYS[1:] + ("type", "language", "series", "male", "female", "tag")
)

# Preparation for future class-based update.
IMAGE_URI_PARTS: list[object] = ["", False, set()]

BASE_DOMAIN: Final[str] = "gold-usergeneratedcontent.net"
RESOURCE_DOMAIN: Final[str] = "ltn." + BASE_DOMAIN
