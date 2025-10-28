"""Tag parsing and fetching helpers."""
from __future__ import annotations

from typing import List, Optional
from urllib.parse import unquote

from .constants import ErrorCode, TAG_TYPES
from .types import StartingCharacter, Tag
from .uri import get_tag_uri
from .utility import HitomiError, fetch


def get_parsed_tags(text: str) -> List[Tag]:
    tags: List[Tag] = []
    raw_positive_tags: set[str] = set()

    text += " "
    current_index = 0
    next_index = text.find(" ")

    while next_index != -1:
        colon_index = text.find(":", current_index)
        if colon_index != -1 and colon_index < next_index:
            is_negative = text.startswith("-", current_index)
            tag_type = text[current_index + (1 if is_negative else 0) : colon_index]
            tag_name = text[colon_index + 1 : next_index]
            tag = Tag(type=tag_type, name=tag_name, is_negative=is_negative)

            if tag.type in TAG_TYPES:
                if tag.name and all(
                    c.isalnum() or c in {"-", "_", "."} for c in tag.name
                ) and tag.name[0].isalnum():
                    positive_raw_tag = f"{tag.type}:{tag.name}"
                    if positive_raw_tag not in raw_positive_tags:
                        tag.name = tag.name.replace("_", " ")
                        tags.append(tag)
                        raw_positive_tags.add(positive_raw_tag)
                    else:
                        raise HitomiError(ErrorCode.DUPLICATED_ELEMENT, f"'{positive_raw_tag}'")
                else:
                    raise HitomiError(
                        ErrorCode.INVALID_VALUE,
                        f"'{tag.name}'",
                        "match /^[a-z0-9][a-z0-9-_.]*$/",
                    )
            else:
                allowed = ", ".join(f"'{value}'" for value in sorted(TAG_TYPES))
                raise HitomiError(ErrorCode.INVALID_VALUE, f"'{tag.type}'", f"be one of {allowed}")
        else:
            snippet = text[current_index:next_index]
            raise HitomiError(ErrorCode.INVALID_VALUE, f"'{snippet}'")

        current_index = next_index + 1
        next_index = text.find(" ", current_index)

    return tags


def get_tags(tag_type: str, starts_with: Optional[StartingCharacter] = None) -> List[Tag]:
    is_type_type = tag_type == "type"
    is_language_type = tag_type == "language"
    has_starts_with = starts_with is not None

    if has_starts_with != (is_type_type or is_language_type):
        if not is_type_type:
            response = fetch(get_tag_uri(tag_type, starts_with)).decode("utf-8")
            tags: List[Tag] = []
            if not is_language_type:
                target = "href=\"/"
                if tag_type in {"male", "female"}:
                    target += f"tag/{tag_type}%3A"
                else:
                    target += f"{tag_type}/"
                end_index = len(target) - 1
                current_index = response.find(target) + len(target)
                next_index = response.find(".", current_index)

                while current_index != end_index:
                    candidate = response[current_index:next_index - 4]
                    if tag_type == "tag":
                        if response.startswith("male", current_index) or response.startswith(
                            "female", current_index
                        ):
                            current_index = response.find(target, next_index) + len(target)
                            next_index = response.find(".", current_index)
                            continue
                    tags.append(Tag(type=tag_type, name=unquote(candidate)))
                    current_index = response.find(target, next_index) + len(target)
                    next_index = response.find(".", current_index)
            else:
                end_index = response.find("}")
                current_index = response.find(":") + 2
                next_index = response.find("\"", current_index)
                while 0 <= next_index < end_index:
                    tags.append(Tag(type="language", name=response[current_index:next_index]))
                    current_index = response.find(":", next_index) + 2
                    next_index = response.find("\"", current_index)
            return tags
        return [
            Tag(type="type", name="doujinshi"),
            Tag(type="type", name="manga"),
            Tag(type="type", name="artistcg"),
            Tag(type="type", name="gamecg"),
            Tag(type="type", name="imageset"),
            Tag(type="type", name="anime"),
        ]
    raise HitomiError(
        ErrorCode.INVALID_VALUE,
        "startsWith",
        "not be used only with type and language",
    )
