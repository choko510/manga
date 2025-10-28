"""Utility helpers used by the Python implementation."""
from __future__ import annotations

import asyncio
import http.client
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, Optional
from urllib.parse import urlsplit

from .constants import BASE_DOMAIN, ErrorCode, RESOURCE_DOMAIN
from .types import IdSet, Node


class HitomiError(Exception):
    """Custom error matching the behaviour of the original library."""

    def __init__(self, code: ErrorCode, *values: str) -> None:
        if code is ErrorCode.INVALID_VALUE:
            message = f"{values[0]} must {'be valid' if len(values) == 1 else values[1]}"
        elif code is ErrorCode.INVALID_CALL:
            message = f"{values[0]} must {values[1]}"
        elif code is ErrorCode.DUPLICATED_ELEMENT:
            message = f"{values[0]} must not be duplicated"
        elif code is ErrorCode.LACK_OF_ELEMENT:
            message = f"{values[0]} must have more elements"
        elif code is ErrorCode.REQUEST_REJECTED:
            escaped_target = values[0].replace("'", "\\'")
            message = f"Request to '{escaped_target}' was rejected"
        else:  # pragma: no cover - defensive programming
            message = "Unknown Hitomi error"
        super().__init__(message)
        self.code = code


_DEFAULT_HEADERS: Dict[str, str] = {
    "Accept": "*/*",
    "Connection": "keep-alive",
    "Referer": f"https://{BASE_DOMAIN}",
}

_FETCH_EXECUTOR = ThreadPoolExecutor()


def _normalise_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    merged = dict(_DEFAULT_HEADERS)
    if headers:
        for key, value in headers.items():
            if key.lower() == "range":
                merged["Range"] = value
            else:
                merged[key.title()] = value
    return merged


def fetch(uri: str, headers: Optional[Dict[str, str]] = None) -> bytes:
    """Fetch a resource over HTTPS and return the raw body."""

    parsed = urlsplit(f"https://{uri}" if "//" not in uri else uri)
    if not parsed.hostname:
        raise HitomiError(ErrorCode.INVALID_VALUE, "uri", "contain a hostname")

    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    connection = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=30)
    try:
        connection.request("GET", path, headers=_normalise_headers(headers))
        response = connection.getresponse()
        status = response.status
        if status not in (200, 206):
            raise HitomiError(ErrorCode.REQUEST_REJECTED, f"https://{uri}")
        data = response.read()
        return data
    finally:
        connection.close()


async def async_fetch(
    uri: str, headers: Optional[Dict[str, str]] = None
) -> bytes:
    """Asynchronous wrapper around :func:`fetch` using a thread executor."""

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_FETCH_EXECUTOR, fetch, uri, headers)


def get_id_set(buffer: bytes, is_negative: bool = False) -> IdSet:
    integers = IdSet(is_negative=is_negative)
    for index in range(0, len(buffer), 4):
        value = int.from_bytes(buffer[index : index + 4], "big", signed=True)
        integers.add(value)
    return integers


def _parse_node(data: bytes) -> Node:
    keys: list[bytes] = []
    datas: list[tuple[int, int]] = []
    subnodes: list[int] = []

    key_count = int.from_bytes(data[0:4], "big", signed=False)
    offset = 4
    for _ in range(key_count):
        key_size = int.from_bytes(data[offset : offset + 4], "big", signed=False)
        if not 0 < key_size < 32:
            raise HitomiError(ErrorCode.INVALID_VALUE, "keySize", "between 1 and 31")
        offset += 4
        keys.append(data[offset : offset + key_size])
        offset += key_size

    data_count = int.from_bytes(data[offset : offset + 4], "big", signed=False)
    offset += 4
    for _ in range(data_count):
        address = int.from_bytes(data[offset : offset + 8], "big", signed=False)
        length = int.from_bytes(data[offset + 8 : offset + 12], "big", signed=True)
        datas.append((address, length))
        offset += 12

    for _ in range(17):
        subnodes.append(int.from_bytes(data[offset : offset + 8], "big", signed=False))
        offset += 8

    return keys, datas, subnodes


@lru_cache(maxsize=256)
def _get_node_bytes(address: int, version: str) -> bytes:
    return fetch(
        f"{RESOURCE_DOMAIN}/galleriesindex/galleries.{version}.index",
        headers={"Range": f"bytes={address}-{address + 463}"},
    )


def get_node_at_address(address: int, version: str) -> Optional[Node]:
    data = _get_node_bytes(address, version)
    if data:
        return _parse_node(data)
    return None


async def async_get_node_at_address(address: int, version: str) -> Optional[Node]:
    data = await asyncio.get_running_loop().run_in_executor(
        _FETCH_EXECUTOR, _get_node_bytes, address, version
    )
    if data:
        return _parse_node(data)
    return None


def binary_search(key: bytes, node: Node, version: str) -> Optional[tuple[int, int]]:
    if not node[0]:
        return None

    compare_result = -1
    index = 0
    keys, data_entries, subnodes = node
    while index < len(keys):
        current_key = keys[index]
        if key < current_key:
            compare_result = -1
        elif key > current_key:
            compare_result = 1
        else:
            compare_result = 0
            break
        if compare_result <= 0:
            break
        index += 1

    if compare_result == 0:
        return data_entries[index]

    child_address = subnodes[index]
    if child_address == 0:
        return None

    if all(address == 0 for address in subnodes):
        return None

    next_node = get_node_at_address(child_address, version)
    if next_node is None:
        return None
    return binary_search(key, next_node, version)


async def async_binary_search(
    key: bytes, node: Node, version: str
) -> Optional[tuple[int, int]]:
    if not node[0]:
        return None

    compare_result = -1
    index = 0
    keys, data_entries, subnodes = node
    while index < len(keys):
        current_key = keys[index]
        if key < current_key:
            compare_result = -1
        elif key > current_key:
            compare_result = 1
        else:
            compare_result = 0
            break
        if compare_result <= 0:
            break
        index += 1

    if compare_result == 0:
        return data_entries[index]

    child_address = subnodes[index]
    if child_address == 0:
        return None

    if all(address == 0 for address in subnodes):
        return None

    next_node = await async_get_node_at_address(child_address, version)
    if next_node is None:
        return None
    return await async_binary_search(key, next_node, version)
