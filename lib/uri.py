"""URI helpers mirroring the behaviour of the original library."""
from __future__ import annotations

import asyncio
import re
from threading import Lock
from typing import Optional
from urllib.parse import quote

from .constants import BASE_DOMAIN, ErrorCode, IMAGE_URI_PARTS, RESOURCE_DOMAIN
from .types import Gallery, Image, PopularityPeriod, StartingCharacter, Tag
from .utility import HitomiError, async_fetch, fetch


def get_nozomi_uri(options: Optional[dict] = None) -> str:
    options = options or {}
    tag: Optional[Tag] = options.get("tag")
    popularity: Optional[PopularityPeriod] = options.get("popularityOrderBy")

    path = "index"
    language = "all"

    if tag is not None:
        tag_type = tag.type
        if tag_type in {"male", "female"}:
            path = f"tag/{tag_type}:{quote(tag.name)}"
        elif tag_type == "language":
            language = tag.name
        else:
            path = f"{tag_type}/{quote(tag.name)}"
    elif popularity is not None:
        path = popularity if popularity != "day" else "today"

    prefix = "popular" if popularity is not None else "n"
    return f"{RESOURCE_DOMAIN}/{prefix}/{path}-{language}.nozomi"


def get_tag_uri(tag_type: str, starts_with: Optional[StartingCharacter] = None) -> str:
    is_language = tag_type == "language"
    has_starts_with = starts_with is not None

    if has_starts_with != is_language:
        subdomain = "ltn." if is_language else ""
        path = "all"
        if is_language:
            path = "language_support"
        else:
            match tag_type:
                case "tag" | "male" | "female":
                    path += "tags"
                case "artist" | "series" | "character" | "group":
                    path += tag_type
                    if not path.endswith("s"):
                        path += "s"
                case _:
                    raise HitomiError(ErrorCode.INVALID_VALUE, "type")
            suffix = starts_with if starts_with != "0-9" else "123"
            path += f"-{suffix}.html"
        return f"{subdomain}{BASE_DOMAIN}/{path}"
    raise HitomiError(ErrorCode.INVALID_VALUE, "startsWith", "not be used with language")


def get_video_uri(gallery: Gallery) -> str:
    if gallery.type == "anime":
        safe_title = gallery.title.display.lower().replace(" ", "-")
        return f"streaming.{BASE_DOMAIN}/videos/{safe_title}.mp4"
    raise HitomiError(ErrorCode.INVALID_VALUE, "gallery['type']", "be 'anime'")


def get_gallery_uri(gallery: Gallery) -> str:
    title_source = gallery.title.japanese or gallery.title.display
    title_bytes = title_source.encode("utf-8")[:200]
    title_text = title_bytes.decode("utf-8", "ignore")
    encoded_title = quote(title_text, safe="")
    encoded_title = re.sub(r"\(|\)|'|%(2[0235F]|3[CEF]|5[BD]|7[BD])", "-", encoded_title)
    gallery_type = gallery.type if gallery.type != "artistcg" else "cg"
    language_suffix = f"-{quote(gallery.language_name.local, safe='')}" if gallery.language_name.local else ""
    return (
        f"{BASE_DOMAIN}/{gallery_type}/{encoded_title}{language_suffix}-{gallery.id}.html"
    ).lower()


class ImageUriResolver:
    """Resolves image URIs with optional synchronous/asynchronous priming."""

    _lock: Lock = Lock()
    _async_lock: asyncio.Lock | None = None
    _signature: tuple[str, bool, tuple[int, ...]] | None = None

    @classmethod
    def _parse_response(cls, response_text: str) -> tuple[str, bool, set[int]]:
        path_code = ""
        starts_with_a = False
        subdomain_codes: set[int] = set()

        for line in response_text.splitlines():
            if not line:
                continue
            match line[0]:
                case "b":
                    path_code = line[4:-2]
                case "o":
                    starts_with_a = line[4] == "0"
                case "c":
                    subdomain_codes.add(int(line[5:-1]))

        if not path_code or not subdomain_codes:
            raise HitomiError(
                ErrorCode.INVALID_VALUE,
                "ImageUriResolver",
                f"{{ pathCode: '{path_code}', startsWithA: {starts_with_a}, subdomainCodes: {len(subdomain_codes)} }}",
            )

        return path_code, starts_with_a, subdomain_codes

    @classmethod
    def _apply_parts(
        cls, path_code: str, starts_with_a: bool, subdomain_codes: set[int]
    ) -> None:
        IMAGE_URI_PARTS[0] = path_code
        IMAGE_URI_PARTS[1] = starts_with_a
        subdomain_set = IMAGE_URI_PARTS[2]
        assert isinstance(subdomain_set, set)
        subdomain_set.clear()
        subdomain_set.update(subdomain_codes)
        cls._signature = (path_code, starts_with_a, tuple(sorted(subdomain_codes)))

    @classmethod
    def _is_initialised(cls) -> bool:
        signature = cls._signature
        if signature is None:
            return False
        path_code, _, subdomain_codes = signature
        return bool(path_code and subdomain_codes)

    @classmethod
    def synchronize(cls, *, force: bool = False) -> None:
        if not force and cls._is_initialised():
            return

        with cls._lock:
            if not force and cls._is_initialised():
                return
            response_text = fetch(f"{RESOURCE_DOMAIN}/gg.js").decode("utf-8")
            parts = cls._parse_response(response_text)
            cls._apply_parts(*parts)

    @classmethod
    async def async_synchronize(cls, *, force: bool = False) -> None:
        if not force and cls._is_initialised():
            return

        with cls._lock:
            if cls._async_lock is None:
                cls._async_lock = asyncio.Lock()
            lock = cls._async_lock

        assert lock is not None
        async with lock:
            if not force and cls._is_initialised():
                return
            response_text = (await async_fetch(f"{RESOURCE_DOMAIN}/gg.js")).decode(
                "utf-8"
            )
            parts = cls._parse_response(response_text)
            cls._apply_parts(*parts)

    @classmethod
    def clear_cache(cls) -> None:
        """Reset the resolver state so that the next call re-fetches metadata."""

        with cls._lock:
            cls._signature = None
            IMAGE_URI_PARTS[0] = ""
            IMAGE_URI_PARTS[1] = False
            subdomain_set = IMAGE_URI_PARTS[2]
            assert isinstance(subdomain_set, set)
            subdomain_set.clear()

    @staticmethod
    def get_image_uri(
        image: Image,
        extension: str,
        *,
        is_thumbnail: bool = False,
        is_small: bool = False,
    ) -> str:
        subdomain_codes = IMAGE_URI_PARTS[2]
        if not isinstance(subdomain_codes, set) or not subdomain_codes:
            raise HitomiError(
                ErrorCode.INVALID_CALL,
                "ImageUriResolver.get_image_uri()",
                "be called after ImageUriResolver.synchronize()",
            )

        if extension not in {"webp", "avif", "jxl"}:
            raise HitomiError(ErrorCode.INVALID_VALUE, "extension")

        if extension == "webp" and not image.has_webp:
            raise HitomiError(ErrorCode.INVALID_VALUE, "extension")
        if extension == "avif" and not image.has_avif:
            raise HitomiError(ErrorCode.INVALID_VALUE, "extension")
        if extension == "jxl" and not image.has_jxl:
            raise HitomiError(ErrorCode.INVALID_VALUE, "extension")

        image_hash_code = int(image.hash[-1] + image.hash[-3:-1], 16)
        subdomain = extension[0]
        path = ""

        if not is_thumbnail:
            path = f"{IMAGE_URI_PARTS[0]}/{image_hash_code}/{image.hash}"
        else:
            if is_small:
                if extension != "avif":
                    raise HitomiError(ErrorCode.INVALID_VALUE, "options['isSmall']", "be used with avif")
                path = "small"
            path += f"bigtn/{image.hash[-1]}/{image.hash[-3:-1]}/{image.hash}"
            subdomain = "tn"

        starts_with_a = bool(IMAGE_URI_PARTS[1])
        in_set = image_hash_code in subdomain_codes
        suffix = "1" if in_set == starts_with_a else "2"
        return f"{subdomain}{suffix}.{BASE_DOMAIN}/{path}.{extension}"
