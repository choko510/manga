"""Python interface mirroring the original Node.js hitomi library."""
from __future__ import annotations

from .gallery import get_gallery, get_gallery_ids
from .tag import get_parsed_tags, get_tags
from .uri import (
    ImageUriResolver,
    get_gallery_uri,
    get_nozomi_uri,
    get_tag_uri,
    get_video_uri,
)
from .utility import HitomiError

__all__ = [
    "get_gallery",
    "get_gallery_ids",
    "get_parsed_tags",
    "get_tags",
    "get_nozomi_uri",
    "get_tag_uri",
    "get_video_uri",
    "get_gallery_uri",
    "ImageUriResolver",
    "HitomiError",
]
