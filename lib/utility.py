"""Utility helpers used by the Python implementation."""
from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Dict, Optional
from urllib.parse import urlsplit

import httpx

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
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/133.0.0.0 Safari/537.36"
    ),
    "Referer": "https://hitomi.la/",
    "Origin": "https://hitomi.la",
}

# httpx セッション管理
_async_client: Optional[httpx.AsyncClient] = None
_sync_client: Optional[httpx.Client] = None


async def _get_async_client() -> httpx.AsyncClient:
    """Get or create httpx async client."""
    global _async_client
    if _async_client is None or _async_client.is_closed:
        _async_client = httpx.AsyncClient(headers=_DEFAULT_HEADERS, http2=True, timeout=10.0)
    return _async_client


def _get_sync_client() -> httpx.Client:
    """Get or create httpx sync client."""
    global _sync_client
    if _sync_client is None or _sync_client.is_closed:
        _sync_client = httpx.Client(headers=_DEFAULT_HEADERS, timeout=10.0)
    return _sync_client


async def close_session() -> None:
    """Close the httpx clients."""
    global _async_client, _sync_client
    if _async_client is not None and not _async_client.is_closed:
        await _async_client.aclose()
        _async_client = None
    if _sync_client is not None and not _sync_client.is_closed:
        _sync_client.close()
        _sync_client = None


def _normalise_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    merged = dict(_DEFAULT_HEADERS)
    if headers:
        for key, value in headers.items():
            if key.lower() == "range":
                merged["Range"] = value
            else:
                merged[key.title()] = value
    return merged


async def async_fetch(
    uri: str,
    headers: Optional[Dict[str, str]] = None,
    *,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> bytes:
    """Fetch a resource over HTTPS and return the raw body.

    Args:
        uri: URL to fetch
        headers: Optional HTTP headers
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 1.0)
    """
    parsed = urlsplit(f"https://{uri}" if "//" not in uri else uri)
    if not parsed.hostname:
        raise HitomiError(ErrorCode.INVALID_VALUE, "uri", "contain a hostname")

    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    url = f"https://{parsed.hostname}{path}"
    req_headers = _normalise_headers(headers)

    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            client = await _get_async_client()
            response = await client.get(url, headers=req_headers)
            status = response.status_code
            if status not in (200, 206):
                last_error = HitomiError(
                    ErrorCode.REQUEST_REJECTED,
                    f"https://{uri} (status={status})",
                )
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    await asyncio.sleep(wait_time)
                continue
            return response.content
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                await asyncio.sleep(wait_time)

    raise last_error or HitomiError(ErrorCode.REQUEST_REJECTED, f"https://{uri}")


def fetch(
    uri: str,
    headers: Optional[Dict[str, str]] = None,
    *,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> bytes:
    """Synchronous fetch using httpx.Client."""
    parsed = urlsplit(f"https://{uri}" if "//" not in uri else uri)
    if not parsed.hostname:
        raise HitomiError(ErrorCode.INVALID_VALUE, "uri", "contain a hostname")

    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    url = f"https://{parsed.hostname}{path}"
    req_headers = _normalise_headers(headers)

    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            client = _get_sync_client()
            response = client.get(url, headers=req_headers)
            status = response.status_code
            if status not in (200, 206):
                last_error = HitomiError(
                    ErrorCode.REQUEST_REJECTED,
                    f"https://{uri} (status={status})",
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2**attempt))
                continue
            return response.content
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                import time
                time.sleep(retry_delay * (2**attempt))

    raise last_error or HitomiError(ErrorCode.REQUEST_REJECTED, f"https://{uri}")



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


# 非同期キャッシュ用
_node_cache: Dict[tuple[int, str], bytes] = {}


async def _async_get_node_bytes(address: int, version: str) -> bytes:
    """Async version to fetch node bytes with caching."""
    cache_key = (address, version)
    if cache_key in _node_cache:
        return _node_cache[cache_key]

    data = await async_fetch(
        f"{RESOURCE_DOMAIN}/galleriesindex/galleries.{version}.index",
        headers={"Range": f"bytes={address}-{address + 463}"},
    )
    _node_cache[cache_key] = data
    return data


@lru_cache(maxsize=256)
def _get_node_bytes(address: int, version: str) -> bytes:
    return fetch(
        f"{RESOURCE_DOMAIN}/galleriesindex/galleries.{version}.index",
        headers={"Range": f"bytes={address}-{address + 463}"},
    )


async def async_get_node_at_address(address: int, version: str) -> Optional[Node]:
    data = await _async_get_node_bytes(address, version)
    if data:
        return _parse_node(data)
    return None


def get_node_at_address(address: int, version: str) -> Optional[Node]:
    data = _get_node_bytes(address, version)
    if data:
        return _parse_node(data)
    return None


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
