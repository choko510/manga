"""Gallery helpers for the Hitomi.la Python port."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .constants import RAW_GALLERY_KEYS, ErrorCode, RESOURCE_DOMAIN
from .types import (
    Gallery,
    GalleryTranslation,
    IdSet,
    Image,
    LanguageName,
    PopularityPeriod,
    Tag,
    Title,
)
from .uri import get_nozomi_uri
from .utility import HitomiError, binary_search, fetch, get_id_set, get_node_at_address


def _parse_date(value: object) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        candidate = value.strip()
        iso_candidate = candidate.replace(" ", "T")
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(iso_candidate, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(iso_candidate)
        except ValueError:
            return None
    return None


def get_gallery(gallery_id: int) -> Gallery:
    response = fetch(f"{RESOURCE_DOMAIN}/galleries/{gallery_id}.js")
    response_text = response.decode("utf-8")
    json_start = response_text.find("{")
    if json_start == -1:
        raise HitomiError(ErrorCode.INVALID_VALUE, "response", "contain valid JSON")
    response_json: Dict[str, object] = json.loads(response_text[json_start:])

    title_value = response_json.get("title")
    if not isinstance(title_value, str):
        raise HitomiError(ErrorCode.INVALID_VALUE, "response['title']", "be string")
    title = Title(
        display=title_value,
        japanese=(
            str(response_json["japanese_title"]) if isinstance(response_json.get("japanese_title"), str) else None
        ),
    )

    language_name = LanguageName(
        english=str(response_json["language"]) if isinstance(response_json.get("language"), str) else None,
        local=str(response_json["language_localname"]) if isinstance(response_json.get("language_localname"), str) else None,
    )

    gallery_type = response_json.get("type")
    if not isinstance(gallery_type, str):
        raise HitomiError(ErrorCode.INVALID_VALUE, "response['type']", "be string")

    related_values = response_json.get("related")
    related_ids: List[int] = []
    if isinstance(related_values, list):
        for value in related_values:
            try:
                related_ids.append(int(value))
            except (TypeError, ValueError):
                continue

    gallery = Gallery(
        id=gallery_id,
        title=title,
        type=gallery_type,
        language_name=language_name,
        artists=[],
        groups=[],
        series=[],
        characters=[],
        tags=[],
        files=[],
        published_date=_parse_date(response_json.get("datepublished") or response_json.get("date")),
        translations=[],
        related_ids=related_ids,
    )

    for key in RAW_GALLERY_KEYS:
        plural = f"{key}s"
        entries = response_json.get(plural)
        if isinstance(entries, list):
            target_attribute = plural if not plural.startswith("p") else "series"
            target_list: List[str] = getattr(gallery, target_attribute)
            for entry in entries:
                if isinstance(entry, dict) and key in entry:
                    target_list.append(str(entry[key]))

    tags = response_json.get("tags")
    if isinstance(tags, list):
        for entry in tags:
            if isinstance(entry, dict):
                tag_type = "tag"
                if entry.get("male"):
                    tag_type = "male"
                elif entry.get("female"):
                    tag_type = "female"
                tag_name = entry.get("tag")
                gallery.tags.append(Tag(type=tag_type, name=str(tag_name or "")))

    files = response_json.get("files")
    if isinstance(files, list):
        for index, entry in enumerate(files):
            if isinstance(entry, dict):
                gallery.files.append(
                    Image(
                        index=index,
                        hash=str(entry.get("hash", "")),
                        name=str(entry.get("name", "")),
                        has_avif=entry.get("hasavif") == 1,
                        has_webp=entry.get("haswebp", 0) != 0,
                        has_jxl=entry.get("hasjxl") == 1,
                        width=int(entry.get("width", 0)),
                        height=int(entry.get("height", 0)),
                    )
                )

    languages = response_json.get("languages")
    if isinstance(languages, list):
        for entry in languages:
            if isinstance(entry, dict):
                english = entry.get("name")
                local = entry.get("language_localname")
                gallery.translations.append(
                    GalleryTranslation(
                        id=int(entry.get("galleryid", 0)),
                        language_name=LanguageName(
                            english=str(english) if isinstance(english, str) else None,
                            local=str(local) if isinstance(local, str) else None,
                        ),
                    )
                )

    return gallery


def _combine_id_sets(base: IdSet, other: IdSet) -> IdSet:
    for gallery_id in list(base):
        contains = gallery_id in other
        if other.is_negative == contains:
            base.discard(gallery_id)
    return base


def get_gallery_ids(options: Optional[dict] = None) -> List[int]:
    options = options or {}
    title = options.get("title")
    tags: Optional[List[Tag]] = options.get("tags")
    range_options: Optional[dict] = options.get("range")
    popularity: Optional[PopularityPeriod] = options.get("popularityOrderBy")

    is_title_available = isinstance(title, str)
    is_tags_available = bool(tags)
    is_range_available = isinstance(range_options, dict)
    should_slice_result = is_range_available and (is_title_available or is_tags_available)

    version = fetch(f"{RESOURCE_DOMAIN}/galleriesindex/version").decode("utf-8")
    id_sets: List[IdSet] = []

    if popularity is not None or is_range_available or (is_tags_available and getattr(tags[0], "is_negative", False)):
        nozomi_options: dict = {}
        if popularity is not None:
            nozomi_options["popularityOrderBy"] = popularity
        uri = get_nozomi_uri(nozomi_options)
        headers = None
        if is_range_available and not should_slice_result and range_options is not None:
            start = range_options.get("start")
            end = range_options.get("end")
            start_bytes = start * 4 if isinstance(start, int) else 0
            end_bytes = f"{end * 4 - 1}" if isinstance(end, int) else ""
            headers = {"Range": f"bytes={start_bytes}-{end_bytes}"}
        id_sets.append(get_id_set(fetch(uri, headers=headers)))

    if is_title_available:
        processed_title = title.lower() + " "
        current_index = 0
        next_index = processed_title.find(" ")
        root_node = get_node_at_address(0, version)
        if root_node is None:
            raise HitomiError(ErrorCode.LACK_OF_ELEMENT, "galleriesIndex")
        while next_index != -1:
            if next_index - current_index == 0:
                raise HitomiError(
                    ErrorCode.INVALID_VALUE,
                    "options['title']",
                    "not contain continuous or edge space",
                )
            word = processed_title[current_index:next_index]
            key = hashlib.sha256(word.encode("utf-8")).digest()[:4]
            data = binary_search(key, root_node, version)
            if data is not None:
                address, length = data
                response = fetch(
                    f"{RESOURCE_DOMAIN}/galleriesindex/galleries.{version}.data",
                    headers={"Range": f"bytes={address + 4}-{address + length - 1}"},
                )
                id_sets.append(get_id_set(response))
            else:
                id_sets.append(get_id_set(b""))
            current_index = next_index + 1
            next_index = processed_title.find(" ", current_index)

    if is_tags_available and tags is not None:
        for tag in tags:
            uri = get_nozomi_uri({"tag": tag})
            id_sets.append(get_id_set(fetch(uri), is_negative=getattr(tag, "is_negative", False)))

    base_set = get_id_set(fetch(get_nozomi_uri()))
    for extra in id_sets:
        base_set = _combine_id_sets(base_set, extra)

    ids = base_set.to_list()
    if should_slice_result and range_options is not None:
        start = range_options.get("start") if isinstance(range_options.get("start"), int) else None
        end = range_options.get("end") if isinstance(range_options.get("end"), int) else None
        return ids[slice(start, end)]
    return ids
