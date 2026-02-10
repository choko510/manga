import asyncio
import hashlib
import json
import os
import random
import re
import secrets
import shlex
import shutil
import string
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock as ThreadLock
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse

import aiohttp
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import Column, Index, Integer, String, Text, func, select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
from starlette.responses import StreamingResponse

from dlsite_async import DlsiteAPI
from lib import ImageUriResolver

# =========================
# グローバル変数管理
# =========================
class GlobalState:
    """グローバル変数を管理するクラス"""
    def __init__(self):
        self.global_session: Optional[aiohttp.ClientSession] = None
        self.scheduler_task: Optional[asyncio.Task] = None
        self.ranking_scheduler_task: Optional[asyncio.Task] = None
        self.personalization_enabled: bool = True
    
    async def cleanup(self):
        """クリーンアップ処理"""
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                print("同期スケジューラ停止")
        
        if self.ranking_scheduler_task:
            self.ranking_scheduler_task.cancel()
            try:
                await self.ranking_scheduler_task
            except asyncio.CancelledError:
                print("ランキング更新スケジューラ停止")
        
        if self.global_session:
            await self.global_session.close()
            print("Global session closed")

# グローバル状態のインスタンス
global_state = GlobalState()

# =========================
# サムネイルメモリキャッシュ
# =========================
class ThumbnailCache:
    """
    サムネイル画像のLRUメモリキャッシュ
    - 最大サイズ: 300MB
    - 最大エントリ数: 3000
    - 古いものから自動削除
    """
    MAX_SIZE_BYTES = 300 * 1024 * 1024  # 300MB
    MAX_ENTRIES = 3000
    
    def __init__(self):
        self._cache: OrderedDict[str, tuple[bytes, str]] = OrderedDict()  # key -> (data, content_type)
        self._current_size = 0
        self._lock = ThreadLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[tuple[bytes, str]]:
        """キャッシュから取得（LRU更新）"""
        with self._lock:
            if key in self._cache:
                # LRU: アクセスされたら末尾に移動
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None
    
    def put(self, key: str, data: bytes, content_type: str) -> None:
        """キャッシュに保存（容量制限あり）"""
        size = len(data)
        
        # 大きすぎるデータはキャッシュしない（例: 10MB以上の単一画像）
        if size > 10 * 1024 * 1024:
            return
        
        with self._lock:
            # 既存のキーがあれば削除（サイズ更新のため）
            if key in self._cache:
                old_data, _ = self._cache.pop(key)
                self._current_size -= len(old_data)
            
            # 容量制限を超える場合、古いエントリを削除
            while (self._current_size + size > self.MAX_SIZE_BYTES or 
                   len(self._cache) >= self.MAX_ENTRIES) and self._cache:
                oldest_key, (oldest_data, _) = self._cache.popitem(last=False)
                self._current_size -= len(oldest_data)
            
            # 新しいエントリを追加
            self._cache[key] = (data, content_type)
            self._current_size += size
    
    def stats(self) -> dict:
        """キャッシュ統計情報"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "entries": len(self._cache),
                "size_mb": round(self._current_size / (1024 * 1024), 2),
                "max_size_mb": round(self.MAX_SIZE_BYTES / (1024 * 1024), 2),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 2),
            }
    
    def clear(self) -> None:
        """キャッシュをクリア"""
        with self._lock:
            self._cache.clear()
            self._current_size = 0
            self._hits = 0
            self._misses = 0

# サムネイルキャッシュのインスタンス
thumbnail_cache = ThumbnailCache()

# グローバル変数（後方互換性）
global_session: Optional[aiohttp.ClientSession] = None

# =========================
# データベース設定
# =========================
DB_FILE = "sa.db"
TRACKING_DB_FILE = "tracking.db"

# メインデータベース（ギャラリー用）
engine: AsyncEngine = create_async_engine(
    f"sqlite+aiosqlite:///db/{DB_FILE}",
    echo=False,
    connect_args={
        "timeout": 20,
    },
    pool_pre_ping=True,
    pool_recycle=3600,
)

# トラッキング用データベース
tracking_engine: AsyncEngine = create_async_engine(
    f"sqlite+aiosqlite:///db/{TRACKING_DB_FILE}",
    echo=False,
    connect_args={
        "timeout": 20,
    },
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)
TrackingSessionLocal = async_sessionmaker(bind=tracking_engine, expire_on_commit=False)

def get_db_session() -> AsyncSession:
    return SessionLocal()


def get_tracking_db_session() -> AsyncSession:
    return TrackingSessionLocal()

# =========================
# モデル
# =========================
Base = declarative_base()
TrackingBase = declarative_base()

class Gallery(Base):
    __tablename__ = 'galleries'
    gallery_id = Column(Integer, primary_key=True)
    japanese_title = Column(String, index=True)
    tags = Column(Text)         # JSON 文字列
    characters = Column(Text)
    files = Column(Text)
    manga_type = Column(String)
    created_at = Column(String) # ISO8601 文字列想定
    page_count = Column(Integer, nullable=True)
    artists = Column(Text)
    # 注意: Computedではなく通常のINTEGERカラム
    # SQLiteのstrftimeはタイムゾーン付き日付をパースできないため
    # Pythonのバックフィル処理で値を設定する
    created_at_unix = Column(Integer, nullable=True)

    def __repr__(self):
        return f"<Gallery(id={self.gallery_id}, title='{self.japanese_title}')>"

# ---- 新トラッキング: おすすめ精度向上のためのログシステム ----

class UserLog(TrackingBase):
    """
    漫画閲覧ログ。
    ユーザーごと・作品ごとに1レコード。複数回訪問時は累積更新。
    """
    __tablename__ = "user_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    manga_id = Column(Integer, nullable=False, index=True)
    avg_time = Column(Integer, default=0)  # 1ページあたり平均閲覧時間（秒）
    read_pages = Column(Integer, default=0)  # 読んだページ数（累積）
    visit_count = Column(Integer, default=0)  # 訪問回数
    total_duration = Column(Integer, default=0)  # 合計閲覧時間（秒）
    last_viewed_at = Column(String)  # 最終閲覧日時

    __table_args__ = (
        Index('ix_user_logs_user_manga', 'user_id', 'manga_id', unique=True),
    )

class Impression(TrackingBase):
    """
    インプレッション・クリックログ（CTR計算用）。
    ユーザーごと・作品ごとに1レコード。
    """
    __tablename__ = "impressions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    manga_id = Column(Integer, nullable=False, index=True)
    shown_count = Column(Integer, default=0)  # 表示回数
    click_count = Column(Integer, default=0)  # クリック回数
    last_shown_at = Column(String)  # 最終表示日時
    last_clicked_at = Column(String)  # 最終クリック日時

    __table_args__ = (
        Index('ix_impressions_user_manga', 'user_id', 'manga_id', unique=True),
    )

class TagPreference(TrackingBase):
    """
    タグ別CTR傾向。
    ユーザーごと・タグごとに1レコード。
    """
    __tablename__ = "tag_preferences"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    tag = Column(String, nullable=False, index=True)
    shown_count = Column(Integer, default=0)  # そのタグ作品の表示回数
    click_count = Column(Integer, default=0)  # そのタグ作品のクリック回数
    total_view_time = Column(Integer, default=0)  # そのタグ作品の合計閲覧時間

    __table_args__ = (
        Index('ix_tag_preferences_user_tag', 'user_id', 'tag', unique=True),
    )

class SearchHistory(TrackingBase):
    """
    検索タグ履歴。
    ユーザーごと・タグごとに1レコード。
    """
    __tablename__ = "search_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    tag = Column(String, nullable=False, index=True)
    search_count = Column(Integer, default=0)  # 検索回数
    last_searched_at = Column(String)  # 最終検索日時

    __table_args__ = (
        Index('ix_search_history_user_tag', 'user_id', 'tag', unique=True),
    )

class UserSnapshot(TrackingBase):
    __tablename__ = 'user_snapshots'

    code = Column(String, primary_key=True)
    payload = Column(Text, nullable=False)
    created_at = Column(String, nullable=False)
    expires_at = Column(String, nullable=False)
    last_accessed = Column(String, nullable=False)

# ---- ランキング ----
class GalleryRanking(Base):
    __tablename__ = 'gallery_rankings'
    gallery_id = Column(Integer, primary_key=True)
    ranking_type = Column(String, primary_key=True)  # 'daily', 'weekly', 'monthly', 'all_time'
    score = Column(Integer, nullable=False, default=0)  # ランキングスコア（アクセス数など）
    view_count = Column(Integer, nullable=False, default=0)  # 閲覧数
    last_updated = Column(String)  # 最終更新日時
    created_at = Column(String)  # 作成日時

# =========================
# FastAPI
# =========================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")




# =========================
# HTTP ヘッダ
# =========================
DEFAULT_CLIENT_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://hitomi.la/",
}

def _build_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = DEFAULT_CLIENT_HEADERS.copy()
    if extra:
        headers.update(extra)
    return headers

# =========================
# ユーティリティ
# =========================
TOKEN_SPLIT_RE = re.compile(r"[ \t\u3000\-\_/\.]+")
_STATIC_FILE_CACHE: Dict[str, Tuple[float, str]] = {}

TAG_TRANSLATIONS_FILE = Path("static/tag-translations.json")
TAG_TRANSLATIONS_HISTORY_DIR = Path("data/tag-translations-history")
TAG_TRANSLATIONS_VERSIONS_FILE = TAG_TRANSLATIONS_HISTORY_DIR / "versions.json"
TAG_TRANSLATIONS_HISTORY_LIMIT = 100
TAG_CATEGORIES_FILE = Path("static/tag-categories.json")
SNAPSHOT_EXPIRY_DAYS = 30


def _generate_snapshot_code(length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def _normalise_tag(value: str) -> str:
    return (value or "").strip().lower()


def _sanitize_snapshot_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(history, list):
        return []
    sanitized: List[Dict[str, Any]] = []
    for entry in history[:200]:
        if isinstance(entry, Mapping):
            sanitized.append(dict(entry))
    return sanitized


def _sanitize_string_list(values: List[str]) -> List[str]:
    if not isinstance(values, list):
        return []
    result: List[str] = []
    for item in values:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                result.append(cleaned)
        elif isinstance(item, (int, float)):
            result.append(str(item))
    return result[:200]


def _sanitize_tag_usage(usage: Dict[str, int]) -> Dict[str, int]:
    if not isinstance(usage, Mapping):
        return {}
    sanitized: Dict[str, int] = {}
    for key, value in usage.items():
        normalised = _normalise_tag(str(key))
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = 0
        if normalised and count > 0:
            sanitized[normalised] = count
    if len(sanitized) <= 200:
        return sanitized
    sorted_items = sorted(sanitized.items(), key=lambda item: item[1], reverse=True)[:200]
    return {key: value for key, value in sorted_items}


def _extract_dlsite_id(q: str) -> Optional[int]:
    """RJ123456, BJ123456 などの作品コードから数値部分を抽出する。"""
    if not q:
        return None
    # 一般的なDLsiteコード形式: [A-Z]{2,}\d+
    match = re.search(r'(?i)(?:RJ|BJ|VJ|RE|OR|RG|CJ|WJ)(\d+)', q)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, TypeError):
            pass
    
    # 数値のみの場合もIDとして扱う（作品コード直接入力の可能性）
    if q.isdigit() and 4 < len(q) < 10:
        return int(q)
    
    return None


def _extract_dlsite_product_code(q: str) -> Optional[str]:
    """URLや文字列からRJ123456などのプロダクトコードを抜き出す。"""
    if not q:
        return None
    # http または https または /dlsite.com/ が含まれる場合はURLとして扱う
    if q.startswith(("http://", "https://")) or "dlsite.com" in q:
        match = re.search(r'(?i)((?:RJ|BJ|VJ|RE|OR|RG|CJ|WJ)\d+)', q)
        if match:
            return match.group(1).upper()
    return None


class TagTranslationsState:
    def __init__(self) -> None:
        self._version: Optional[str] = None
        self._condition = asyncio.Condition()

    async def get_version(self) -> Optional[str]:
        if self._version is None:
            self._version = await _load_current_tag_translation_version()
        return self._version

    async def set_version(self, version: str) -> None:
        self._version = version
        async with self._condition:
            self._condition.notify_all()

    async def wait_for_update(self, since: Optional[str], timeout: float = 30.0) -> Tuple[Optional[str], bool]:
        current = await self.get_version()
        if current != since:
            return current, True
        try:
            async with self._condition:
                await asyncio.wait_for(self._condition.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        current = await self.get_version()
        return current, current != since


tag_translations_state = TagTranslationsState()
tag_translations_lock = asyncio.Lock()


def _sanitize_alias_list(values: Any) -> List[str]:
    if isinstance(values, str):
        candidates = [values]
    elif isinstance(values, list):
        candidates = values
    else:
        return []

    result: List[str] = []
    seen: Set[str] = set()
    for item in candidates:
        if isinstance(item, (int, float)):
            candidate = str(item)
        elif isinstance(item, str):
            candidate = item
        else:
            continue
        cleaned = candidate.strip()
        if not cleaned:
            continue
        normalised = _normalise_tag(cleaned)
        if not normalised or normalised in seen:
            continue
        seen.add(normalised)
        result.append(cleaned)
    return result[:50]


def _normalize_translation_entry(value: Any) -> Dict[str, Any]:
    translation = ""
    description = ""
    priority = 0
    aliases: List[str] = []
    if isinstance(value, Mapping):
        translation = str(value.get("translation", "")).strip()
        description = str(value.get("description", "")).strip()
        aliases = _sanitize_alias_list(value.get("aliases", []))
        try:
            priority = int(value.get("priority", 0))
        except (ValueError, TypeError):
            priority = 0
    elif isinstance(value, str):
        translation = value.strip()
    return {
        "translation": translation,
        "description": description,
        "priority": priority,
        "aliases": aliases,
    }


def _generate_tag_translation_version_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")


async def _load_tag_translation_versions() -> List[Dict[str, Any]]:
    raw = await _read_json_file(TAG_TRANSLATIONS_VERSIONS_FILE, [])
    if not isinstance(raw, list):
        return []
    versions: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        version = str(item.get("version", "")).strip()
        created_at = str(item.get("created_at", "")).strip()
        reason = str(item.get("reason", "")).strip()
        auto = bool(item.get("auto", False))
        restored_from = str(item.get("restored_from", "")).strip()
        parent_version = str(item.get("parent_version", "")).strip()
        if not version:
            continue
        entry: Dict[str, Any] = {"version": version}
        if created_at:
            entry["created_at"] = created_at
        if reason:
            entry["reason"] = reason
        if parent_version:
            entry["parent_version"] = parent_version
        if restored_from:
            entry["restored_from"] = restored_from
        if auto:
            entry["auto"] = True
        versions.append(entry)
    versions.sort(key=lambda item: item.get("created_at", "") or item["version"])
    return versions[-TAG_TRANSLATIONS_HISTORY_LIMIT:]


async def _write_tag_translation_versions(entries: List[Dict[str, Any]]) -> None:
    await _write_json_file(TAG_TRANSLATIONS_VERSIONS_FILE, entries, sort_keys=True)


async def _load_current_tag_translation_version() -> Optional[str]:
    versions = await _load_tag_translation_versions()
    if not versions:
        return None
    return versions[-1].get("version")


async def _record_tag_translation_version(
    data: Dict[str, Dict[str, Any]],
    *,
    reason: str,
    auto: bool,
    parent_version: Optional[str] = None,
    restored_from: Optional[str] = None,
) -> Dict[str, Any]:
    TAG_TRANSLATIONS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    version_id = _generate_tag_translation_version_id()
    timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    entry: Dict[str, Any] = {
        "version": version_id,
        "created_at": timestamp,
        "reason": reason,
    }
    if auto:
        entry["auto"] = True
    if parent_version:
        entry["parent_version"] = parent_version
    if restored_from:
        entry["restored_from"] = restored_from

    versions = await _load_tag_translation_versions()
    versions.append(entry)
    versions = versions[-TAG_TRANSLATIONS_HISTORY_LIMIT:]

    snapshot_path = TAG_TRANSLATIONS_HISTORY_DIR / f"{version_id}.json"
    await _write_json_file(snapshot_path, data, sort_keys=True)
    await _write_tag_translation_versions(versions)
    await tag_translations_state.set_version(version_id)
    return entry


async def _ensure_tag_translation_history_initialized(
    data: Dict[str, Dict[str, Any]]
) -> Optional[str]:
    versions = await _load_tag_translation_versions()
    if versions:
        return versions[-1].get("version")
    entry = await _record_tag_translation_version(
        data,
        reason="initial",
        auto=False,
    )
    return entry.get("version")

# ランキングデータファイル
RANKING_FILES = {
    "daily": "day_ids.txt",
    "weekly": "week_ids.txt",
    "monthly": "month_ids.txt",
    "yearly": "year_ids.txt",
}

# ランキングIDのキャッシュ (60秒保持)
_RANKING_CACHE: Dict[str, Tuple[float, List[int]]] = {}
_RANKING_CACHE_TTL = 60.0  # 秒
_RANKING_CACHE_LOCK = asyncio.Lock()

# ページ数条件に該当するギャラリーIDのセット（永続キャッシュ - 再起動までメモリ保持）
# キー: (min_pages, max_pages) -> セット of gallery_ids
_PAGE_RANGE_GALLERY_IDS_CACHE: Dict[Tuple[int, int], Set[int]] = {}

# 検索結果COUNTクエリキャッシュ（永続キャッシュ - 再起動までメモリ保持）
# キー: (title, tag, exclude_tag, character, min_pages, max_pages) のハッシュ
_SEARCH_COUNT_CACHE: Dict[str, int] = {}

# 検索結果キャッシュ（TTL付き - 600秒保持）
# キー: 検索条件のハッシュ -> (timestamp, results, total_count)
_SEARCH_RESULTS_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]], int]] = {}
_SEARCH_RESULTS_CACHE_TTL = 600.0  # 秒
_SEARCH_RESULTS_CACHE_MAX_SIZE = 100  # キャッシュの最大エントリ数


def _make_search_results_cache_key(
    title: Optional[str],
    tag: Optional[str],
    exclude_tag: Optional[str],
    character: Optional[str],
    q: Optional[str],
    limit: int,
    offset: int,
    after_created_at: Optional[str],
    after_gallery_id: Optional[int],
    min_pages: Optional[int],
    max_pages: Optional[int],
    sort_by: Optional[str],
) -> str:
    """検索条件からキャッシュキーを生成"""
    key_parts = [
        title or "",
        tag or "",
        exclude_tag or "",
        character or "",
        q or "",
        str(limit),
        str(offset),
        after_created_at or "",
        str(after_gallery_id) if after_gallery_id is not None else "",
        str(min_pages) if min_pages is not None else "",
        str(max_pages) if max_pages is not None else "",
        sort_by or "",
    ]
    return hashlib.md5("|".join(key_parts).encode()).hexdigest()


def _get_cached_search_results(cache_key: str) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    """キャッシュから検索結果を取得（TTLチェック付き）"""
    cached = _SEARCH_RESULTS_CACHE.get(cache_key)
    if cached is None:
        return None
    timestamp, results, total_count = cached
    if time.time() - timestamp > _SEARCH_RESULTS_CACHE_TTL:
        # TTL切れ: キャッシュから削除
        _SEARCH_RESULTS_CACHE.pop(cache_key, None)
        return None
    return results, total_count


def _set_cached_search_results(cache_key: str, results: List[Dict[str, Any]], total_count: int) -> None:
    """検索結果をキャッシュに保存（サイズ制限付き）"""
    # キャッシュサイズが上限を超えた場合、古いエントリを削除
    if len(_SEARCH_RESULTS_CACHE) >= _SEARCH_RESULTS_CACHE_MAX_SIZE:
        # 最も古いエントリを削除
        oldest_key = min(_SEARCH_RESULTS_CACHE.keys(), key=lambda k: _SEARCH_RESULTS_CACHE[k][0])
        _SEARCH_RESULTS_CACHE.pop(oldest_key, None)
    _SEARCH_RESULTS_CACHE[cache_key] = (time.time(), results, total_count)

async def _get_page_range_gallery_ids(min_pages: int, max_pages: int) -> Set[int]:
    """
    指定されたページ数範囲に該当するギャラリーIDのセットを取得（永続キャッシュ）。
    ランキング+ページ数フィルターの高速化に使用。
    """
    cache_key = (min_pages, max_pages)
    
    # キャッシュにあればそのまま返す（TTLチェックなし）
    if cache_key in _PAGE_RANGE_GALLERY_IDS_CACHE:
        return _PAGE_RANGE_GALLERY_IDS_CACHE[cache_key]
    
    # キャッシュミス: DBから取得
    async with get_db_session() as db:
        query = """
            SELECT gallery_id FROM galleries
            WHERE manga_type IN ('doujinshi', 'manga')
            AND page_count BETWEEN :min_pages AND :max_pages
        """
        result = await db.execute(text(query), {"min_pages": min_pages, "max_pages": max_pages})
        ids = {row.gallery_id for row in result.fetchall()}
    
    _PAGE_RANGE_GALLERY_IDS_CACHE[cache_key] = ids
    return ids

def _make_search_count_cache_key(
    title: Optional[str],
    tag: Optional[str],
    exclude_tag: Optional[str],
    character: Optional[str],
    q: Optional[str],
    min_pages: Optional[int],
    max_pages: Optional[int],
) -> str:
    """検索条件からキャッシュキーを生成"""
    key_parts = [
        title or "",
        tag or "",
        exclude_tag or "",
        character or "",
        q or "",
        str(min_pages) if min_pages is not None else "",
        str(max_pages) if max_pages is not None else "",
    ]
    return hashlib.md5("|".join(key_parts).encode()).hexdigest()

async def _get_search_count(
    title: Optional[str],
    tag: Optional[str],
    exclude_tag: Optional[str],
    character: Optional[str],
    q: Optional[str],
    min_pages: Optional[int],
    max_pages: Optional[int],
    known_tags: Set[str],
) -> int:
    """
    検索条件に基づく総件数を取得する（永続キャッシュ）。
    独自のDBセッションを使用することで、検索クエリと並列実行可能。
    """
    cache_key = _make_search_count_cache_key(
        title, tag, exclude_tag, character, q, min_pages, max_pages
    )
    
    # キャッシュにあればそのまま返す（TTLチェックなし）
    if cache_key in _SEARCH_COUNT_CACHE:
        return _SEARCH_COUNT_CACHE[cache_key]
    
    # キャッシュミス: 独自のセッションでDBからCOUNT取得
    async with get_db_session() as db_session:
        count_query = """
            SELECT COUNT(*) as total_count
            FROM galleries AS g
            WHERE g.manga_type IN ('doujinshi', 'manga')
        """
        
        count_params: Dict[str, Any] = {}
        count_where_clauses: List[str] = []
        
        if title:
            count_where_clauses.append("g.japanese_title LIKE :title_val")
            count_params["title_val"] = f"%{title}%"
        
        if character:
            count_where_clauses.append("g.characters LIKE :character_val")
            count_params["character_val"] = f"%{character}%"
        
        if tag:
            tag_terms = _parse_tag_terms(tag, known_tags)
            if tag_terms:
                exists_clauses, exists_params = _build_tag_exists_clause("g", tag_terms)
                count_where_clauses.extend(exists_clauses)
                count_params.update(exists_params)
        
        if exclude_tag:
            exclude_tag_terms = _parse_tag_terms(exclude_tag, known_tags)
            if exclude_tag_terms:
                not_exists_clauses, not_exists_params = _build_tag_not_exists_clause("g", exclude_tag_terms)
                count_where_clauses.extend(not_exists_clauses)
                count_params.update(not_exists_params)
        
        if q:
            q_clauses = []
            q_clauses.append("g.japanese_title LIKE :q_like")
            q_clauses.append("g.characters LIKE :q_like")
            
            # タグとしてのチェック
            q_tag = q.lower()
            q_artist = f"artist:{q_tag}"
            
            # EXISTS句でのタグチェック
            q_clauses.append(f"EXISTS (SELECT 1 FROM gallery_tags WHERE gallery_id = g.gallery_id AND tag IN (:q_tag, :q_artist))")
            
            # 作品コードのチェック
            code_id = _extract_dlsite_id(q)
            if code_id:
                q_clauses.append("g.gallery_id = :code_id")
                count_params["code_id"] = code_id

            count_where_clauses.append("(" + " OR ".join(q_clauses) + ")")
            count_params["q_like"] = f"%{q}%"
            count_params["q_tag"] = q_tag
            count_params["q_artist"] = q_artist
        
        if min_pages is not None or max_pages is not None:
            min_val = max(min_pages or 0, 0)
            max_val = max_pages if max_pages is not None else 10_000
            if max_val < min_val:
                max_val = min_val
            count_params["min_pages"] = min_val
            count_params["max_pages"] = max_val
            count_where_clauses.append("g.page_count BETWEEN :min_pages AND :max_pages")
        
        if count_where_clauses:
            count_query += " AND " + " AND ".join(count_where_clauses)
        
        count_result = await db_session.execute(text(count_query), count_params)
        total_count = count_result.scalar() or 0
    
    # キャッシュに保存（永続）
    _SEARCH_COUNT_CACHE[cache_key] = total_count
    return total_count


async def _load_ranking_ids(ranking_type: str) -> List[int]:
    """
    ランキング情報をデータベースまたはファイルから読み込む（キャッシュ付き）
    ranking_type: 'daily', 'weekly', 'monthly', 'yearly'
    """
    if ranking_type not in RANKING_FILES and ranking_type != 'all_time':
        raise ValueError(f"Invalid ranking type: {ranking_type}")

    # キャッシュをチェック（ロックなし高速パス）
    now = time.time()
    cached = _RANKING_CACHE.get(ranking_type)
    if cached and now - cached[0] < _RANKING_CACHE_TTL:
        return cached[1]

    async with _RANKING_CACHE_LOCK:
        # ロック内で再チェック
        now = time.time()
        cached = _RANKING_CACHE.get(ranking_type)
        if cached and now - cached[0] < _RANKING_CACHE_TTL:
            return cached[1]

        async def _read_from_db() -> List[int]:
            try:
                async with get_db_session() as db:
                    stmt = text(
                        """
                        SELECT gallery_id
                        FROM gallery_rankings
                        WHERE ranking_type = :ranking_type
                        ORDER BY score DESC, view_count DESC, last_updated DESC, gallery_id DESC
                        """
                    )
                    result = await db.execute(stmt, {"ranking_type": ranking_type})
                    rows = result.fetchall()
            except Exception as exc:
                print(f"ランキングデータのDB読込に失敗: {exc}")
                return []

            ids: List[int] = []
            for row in rows:
                try:
                    gallery_id = (
                        row[0]
                        if isinstance(row, (list, tuple))
                        else row.gallery_id
                        if hasattr(row, "gallery_id")
                        else None
                    )
                    if isinstance(gallery_id, int):
                        ids.append(gallery_id)
                except Exception:
                    continue
            return ids

        db_ids = await _read_from_db()
        if db_ids:
            _RANKING_CACHE[ranking_type] = (now, db_ids)
            return db_ids

        file_path = RANKING_FILES.get(ranking_type, "")
        if not file_path:
            _RANKING_CACHE[ranking_type] = (now, [])
            return []

        def _read_ids() -> List[int]:
            try:
                with open("cache/"+file_path, 'r', encoding='utf-8') as f:
                    return [int(line.strip()) for line in f if line.strip().isdigit()]
            except FileNotFoundError:
                print(f"ランキングファイルが見つかりません: {file_path}")
                return []
            except Exception as e:
                print(f"ランキングファイル読み込みエラー: {e}")
                return []

        file_ids = await asyncio.to_thread(_read_ids)
        _RANKING_CACHE[ranking_type] = (now, file_ids)
        return file_ids


def _load_static_file(path: str) -> str:
    """静的ファイルをキャッシュして返す"""
    file_path = Path(path)
    stat = file_path.stat()
    cached = _STATIC_FILE_CACHE.get(path)
    if cached and cached[0] == stat.st_mtime:
        return cached[1]

    content = file_path.read_text(encoding="utf-8")
    _STATIC_FILE_CACHE[path] = (stat.st_mtime, content)
    return content

def _serve_cached_html(path: str) -> HTMLResponse:
    try:
        content = _load_static_file(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"{path} not found") from exc
    return HTMLResponse(content=content)


async def _read_json_file(path: Path, default: Any) -> Any:
    def _load() -> Any:
        if not path.exists():
            return default
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            return default
        return json.loads(content)

    try:
        return await asyncio.to_thread(_load)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid JSON format in {path.name}") from exc


async def _write_json_file(path: Path, data: Any, *, sort_keys: bool = False) -> None:
    def _dump() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(data, ensure_ascii=False, indent=4, sort_keys=sort_keys)
        path.write_text(text + "\n", encoding="utf-8")

    await asyncio.to_thread(_dump)

class MultipartDownloadError(Exception):
    """Raised when multi-part download cannot be completed."""


# =========================
# 画像プロキシ高速化用キャッシュ
# =========================
class ImageProxyCache:
    """
    LRUベースのインメモリ画像キャッシュ
    - 小さい画像（サムネイル等）を高速に返すためのキャッシュ
    - サイズ制限と有効期限でメモリ使用量を制御
    """
    def __init__(self, max_size_bytes: int = 150 * 1024 * 1024, max_item_size: int = 512 * 1024, ttl_seconds: int = 600):
        self._cache: Dict[str, Tuple[bytes, str, float]] = {}  # {key: (data, content_type, timestamp)}
        self._access_order: List[str] = []  # LRU tracking
        self._current_size: int = 0
        self._max_size = max_size_bytes
        self._max_item_size = max_item_size  # 個別アイテムの最大サイズ（これ以上はキャッシュしない）
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, url: str) -> str:
        """URLからキャッシュキーを生成"""
        return hashlib.md5(url.encode()).hexdigest()
    
    async def get(self, url: str) -> Optional[Tuple[bytes, str]]:
        """キャッシュから取得。ヒットしたらデータとcontent_typeを返す"""
        key = self._make_key(url)
        async with self._lock:
            if key in self._cache:
                data, content_type, ts = self._cache[key]
                # TTLチェック
                if time.time() - ts > self._ttl:
                    # 期限切れ
                    self._evict_key(key)
                    self._misses += 1
                    return None
                # LRU更新
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                self._hits += 1
                return data, content_type
            self._misses += 1
            return None
    
    async def set(self, url: str, data: bytes, content_type: str) -> None:
        """キャッシュに保存（サイズ制限を超えるアイテムはスキップ）"""
        if len(data) > self._max_item_size:
            return  # 大きすぎるアイテムはキャッシュしない
        
        key = self._make_key(url)
        async with self._lock:
            # 既存エントリを削除
            if key in self._cache:
                self._evict_key(key)
            
            # スペースを確保
            while self._current_size + len(data) > self._max_size and self._access_order:
                oldest_key = self._access_order[0]
                self._evict_key(oldest_key)
            
            # 追加
            self._cache[key] = (data, content_type, time.time())
            self._access_order.append(key)
            self._current_size += len(data)
    
    def _evict_key(self, key: str) -> None:
        """キーを削除（ロック保持前提）"""
        if key in self._cache:
            data, _, _ = self._cache.pop(key)
            self._current_size -= len(data)
        if key in self._access_order:
            self._access_order.remove(key)
    
    def stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cached_items": len(self._cache),
            "current_size_mb": round(self._current_size / 1024 / 1024, 2),
            "max_size_mb": round(self._max_size / 1024 / 1024, 2),
        }

# グローバルキャッシュインスタンス
_image_proxy_cache = ImageProxyCache()


# =========================
# DB 初期化
# =========================

def _parse_page_count_from_files(files_json: str) -> Optional[int]:
    """filesのJSONからpage_countを算出（スレッドプール向け）"""
    try:
        if not files_json:
            return None
        data = json.loads(files_json)
        if not isinstance(data, list):
            return None
        count = sum(
            1 for item in data
            if isinstance(item, dict)
            and isinstance(item.get("hash"), str)
            and item.get("hash")
        )
        return count
    except Exception:
        return None


def _parse_created_at_to_unix(created_at: str) -> Optional[int]:
    """created_at文字列をUNIXタイムスタンプに変換（スレッドプール向け）"""
    try:
        if not created_at:
            return None
        s = str(created_at).strip()
        if not s:
            return None
        iso = s.replace(" ", "T")
        # 最も一般的なフォーマットを最初に試す
        dt = None
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(iso, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            try:
                dt = datetime.fromisoformat(iso)
            except ValueError:
                return None
        if dt is None:
            return None
        unix_ts = int(dt.timestamp())
        return unix_ts if unix_ts > 0 else None
    except Exception:
        return None


def _process_page_count_batch(rows: List[Tuple[int, str]]) -> Tuple[List[Dict[str, int]], int]:
    """バッチ内の全行のpage_countを計算（スレッドプール用）"""
    updates = []
    errors = 0
    for gallery_id, files_json in rows:
        count = _parse_page_count_from_files(files_json)
        if count is not None:
            updates.append({"gid": gallery_id, "cnt": count})
        elif files_json:  # files_jsonがあるのにパースできなかった場合
            errors += 1
    return updates, errors


def _process_created_at_batch(rows: List[Tuple[int, str]]) -> Tuple[List[Dict[str, int]], int]:
    """バッチ内の全行のcreated_at_unixを計算（スレッドプール用）"""
    updates = []
    errors = 0
    for gallery_id, created_at in rows:
        unix_ts = _parse_created_at_to_unix(created_at)
        if unix_ts is not None:
            updates.append({"gid": gallery_id, "ts": unix_ts})
        elif created_at:  # created_atがあるのにパースできなかった場合
            errors += 1
    return updates, errors


def _batch_parse_files_json(results: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """検索結果のfilesフィールドをバッチでJSONパースする（スレッドプール用）"""
    parsed_files = []
    for r in results:
        try:
            files_str = r.get("files")
            if isinstance(files_str, str):
                files_data = json.loads(files_str)
                files_list = files_data if isinstance(files_data, list) else []
            elif isinstance(files_str, list):
                files_list = files_str
            else:
                files_list = []
        except (json.JSONDecodeError, TypeError):
            files_list = []
        parsed_files.append(files_list)
    return parsed_files


async def _process_results_with_image_urls(results: List[Dict[str, Any]]) -> None:
    """
    検索結果にimage_urlsを追加し、filesキーを削除する共通処理。
    results引数はin-placeで変更される。
    """
    if not results:
        return
    
    # 1. JSONパースをスレッドプールで実行
    files_data_cache = await asyncio.to_thread(_batch_parse_files_json, results)
    
    # 2. geturl タスクを作成して並列実行
    geturl_tasks = [
        geturl({"gallery_id": r["gallery_id"], "files": files_data_cache[i]})
        for i, r in enumerate(results)
    ]
    image_urls_results = await asyncio.gather(*geturl_tasks, return_exceptions=True)
    
    # 3. 結果をマージ
    for i, r in enumerate(results):
        image_urls = image_urls_results[i]
        if isinstance(image_urls, Exception):
            r["image_urls"] = []
        else:
            r["image_urls"] = image_urls
        
        files_list = files_data_cache[i]
        stored_pages = r.get("page_count")
        if not isinstance(stored_pages, int) or stored_pages < 0:
            r["page_count"] = len(files_list)
        
        if "files" in r:
            del r["files"]


async def _backfill_database_data():
    """
    バックグラウンドで実行するデータ補完処理
    - page_count: files(JSON配列)中の hash を数える
    - created_at_unix: created_at(ISO8601想定) をUNIX秒に変換して保存
    
    最適化:
    - バッチサイズを5000に増加
    - JSONパースと日付パースをスレッドプールで並列実行
    - executemanyで一括UPDATE
    """
    print("[backfill] === バックグラウンドデータ補完処理を開始 ===")
    start_time = time.time()
    BATCH_SIZE = 5000  # バッチサイズを増加
    
    try:
        # 高速チェック: 補完対象が存在するかEXISTSで確認（COUNT(*)よりはるかに高速）
        async with engine.begin() as conn:
            # page_count補完対象の存在チェック
            page_count_needs_backfill = await conn.execute(text(
                "SELECT EXISTS(SELECT 1 FROM galleries WHERE page_count IS NULL AND files IS NOT NULL LIMIT 1)"
            ))
            has_page_count_work = page_count_needs_backfill.scalar()
            
            # created_at_unix列の存在確認とbackfill対象チェック
            pragma_res = await conn.execute(text("PRAGMA table_info(galleries)"))
            gallery_columns = [row[1] for row in pragma_res.fetchall()]
            has_created_at_unix_column = "created_at_unix" in gallery_columns
            
            has_created_at_work = False
            if has_created_at_unix_column:
                created_at_needs_backfill = await conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM galleries WHERE (created_at_unix IS NULL OR created_at_unix = 0) AND created_at IS NOT NULL LIMIT 1)"
                ))
                has_created_at_work = created_at_needs_backfill.scalar()
        
        # 補完対象がなければ即座に終了
        if not has_page_count_work and not has_created_at_work:
            elapsed = time.time() - start_time
            print(f"[backfill] page_count: 補完対象なし（全件処理済み）")
            print(f"[backfill] created_at_unix: 補完対象なし（全件処理済み）")
            print(f"[backfill] === バックグラウンドデータ補完処理完了 ({elapsed:.1f}秒) ===")
            return
        
        # 3-1. page_count が NULL の行を補完（バッチごとにコミット）
        total_page_count_updated = 0
        total_page_count_errors = 0
        batch_num = 0
        
        if has_page_count_work:
            print("[backfill] page_count 補完処理開始...")
            
            while True:
                batch_num += 1
                
                try:
                    async with engine.begin() as conn:
                        result = await conn.execute(
                            text(
                                "SELECT gallery_id, files "
                                "FROM galleries "
                                "WHERE page_count IS NULL "
                                "AND files IS NOT NULL "
                                f"LIMIT {BATCH_SIZE}"
                            )
                        )
                        rows = result.fetchall()
                        
                        if not rows:
                            break
                        
                        # スレッドプールでJSONパースを実行（CPUバウンドな処理をオフロード）
                        updates, batch_errors = await asyncio.to_thread(
                            _process_page_count_batch, rows
                        )
                        
                        # 一括UPDATEで効率化
                        if updates:
                            await conn.execute(
                                text("UPDATE galleries SET page_count = :cnt WHERE gallery_id = :gid"),
                                updates,
                            )
                        
                        total_page_count_updated += len(updates)
                        total_page_count_errors += batch_errors
                        
                        # バッチ完了ログ（エラーがあればサマリー表示）
                        if batch_errors > 0:
                            print(f"[backfill] バッチ {batch_num}: {len(updates)}件更新, {batch_errors}件エラー")
                        elif batch_num % 10 == 0:  # 10バッチごとに進捗表示
                            print(f"[backfill] バッチ {batch_num}: 累計 {total_page_count_updated}件更新")
                            
                except Exception as batch_err:
                    print(f"[backfill] バッチ {batch_num} 致命的エラー: {type(batch_err).__name__}: {batch_err}")
                    import traceback
                    traceback.print_exc()
                    break
                    
                # バッチ処理後に短い遅延を入れて他の処理に譲る
                await asyncio.sleep(0.001)
            
            if total_page_count_updated > 0 or total_page_count_errors > 0:
                print(f"[backfill] page_count 補完完了: 成功 {total_page_count_updated}件, エラー {total_page_count_errors}件")
        else:
            print("[backfill] page_count: 補完対象なし（全件処理済み）")

        # 3-2. created_at_unix 列が存在する場合のみ補完処理を行う
        total_created_at_unix_updated = 0
        total_created_at_unix_errors = 0
        
        if has_created_at_work:
            print("[backfill] created_at_unix 補完処理開始...")
            batch_num = 0
            
            while True:
                batch_num += 1
                
                async with engine.begin() as conn:
                    result = await conn.execute(
                        text(
                            "SELECT gallery_id, created_at "
                            "FROM galleries "
                            "WHERE (created_at_unix IS NULL OR created_at_unix = 0) "
                            "AND created_at IS NOT NULL "
                            f"LIMIT {BATCH_SIZE}"
                        )
                    )
                    rows = result.fetchall()
                    
                    if not rows:
                        if batch_num == 1:
                            print("[backfill] created_at_unix: 補完対象なし（全件処理済み）")
                        break
                    
                    # スレッドプールで日付パースを実行
                    updates, batch_errors = await asyncio.to_thread(
                        _process_created_at_batch, rows
                    )
                    
                    # 一括UPDATEで効率化
                    if updates:
                        await conn.execute(
                            text("UPDATE galleries SET created_at_unix = :ts WHERE gallery_id = :gid"),
                            updates,
                        )
                    
                    total_created_at_unix_updated += len(updates)
                    total_created_at_unix_errors += batch_errors
                    
                    if batch_num % 10 == 0:
                        print(f"[backfill] created_at_unix バッチ {batch_num}: 累計 {total_created_at_unix_updated}件更新")
                        
                # バッチ処理後に短い遅延を入れて他の処理に譲る
                await asyncio.sleep(0.001)
            
            if total_created_at_unix_updated > 0 or total_created_at_unix_errors > 0:
                print(f"[backfill] created_at_unix 補完完了: 成功 {total_created_at_unix_updated}件, エラー {total_created_at_unix_errors}件")
        else:
            print("[backfill] created_at_unix: 補完対象なし（全件処理済み）")
        
        elapsed = time.time() - start_time
        print(f"[backfill] === バックグラウンドデータ補完処理完了 ({elapsed:.1f}秒) ===")
        
    except Exception as e:
        print(f"[backfill] 致命的エラー: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

async def init_database():
    """
    - 通常テーブル作成
    - SQLite PRAGMA 最適化
    - FTS5 仮想テーブル + 同期トリガー
    - gallery_tags 正規化テーブル + 強インデックス
    - tag_stats 集約テーブル（インクリメンタル更新）
    - 初回/不一致時の REBUILD と BACKFILL
    
    最適化:
    - SQL文をバッチ実行してオーバーヘッド削減
    - 重複したDROP TRIGGER文を削除
    """

    global engine, SessionLocal
    
    print("[init_db] === データベース初期化開始 ===")
    start_time = time.time()

    # 接続確認
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        print(f"[init_db] 接続エラー、データベース再作成: {type(e).__name__}: {e}")
        await engine.dispose()
        # バックアップして削除
        if os.path.exists(f"db/{DB_FILE}"):
            try:
                shutil.copy2(f"db/{DB_FILE}", f"db/{DB_FILE}.corrupt_backup")
                print(f"[init_db] バックアップ作成: db/{DB_FILE}.corrupt_backup")
            except Exception:
                pass
            for suffix in ["-wal", "-shm"]:
                p = f"db/{DB_FILE}{suffix}"
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            try:
                os.remove(f"db/{DB_FILE}")
                print(f"[init_db] 破損DBファイル削除完了")
            except Exception:
                pass
        engine = create_async_engine(
            f"sqlite+aiosqlite:///db/{DB_FILE}",
            echo=False,
            connect_args={"timeout": 20},
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with engine.begin() as conn:
        # テーブル作成
        await conn.run_sync(Base.metadata.create_all)

        # 不足している列を追加
        res = await conn.execute(text("PRAGMA table_info(galleries)"))
        columns = {row[1] for row in res.fetchall()}
        if "page_count" not in columns:
            await conn.execute(text("ALTER TABLE galleries ADD COLUMN page_count INTEGER"))
        if "created_at_unix" not in columns:
            await conn.execute(text("ALTER TABLE galleries ADD COLUMN created_at_unix INTEGER"))

    print("[init_db] スキーマとインデックス構築中...")
    async with engine.begin() as conn:
        # --- PRAGMA設定（バッチ実行） ---
        pragma_statements = [
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA cache_size=20000",
            "PRAGMA temp_store=MEMORY",
            "PRAGMA foreign_keys=ON",
            "PRAGMA mmap_size=536870912",  # 512MB
        ]
        for stmt in pragma_statements:
            await conn.execute(text(stmt))

        # --- インデックス作成（バッチ実行） ---
        index_statements = [
            "CREATE INDEX IF NOT EXISTS idx_galleries_created ON galleries(created_at DESC, gallery_id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_galleries_type_created_id ON galleries(manga_type, created_at DESC, gallery_id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_galleries_characters ON galleries(characters)",
            # パーシャルインデックス: backfill対象のNULL行を高速検索
            "CREATE INDEX IF NOT EXISTS idx_galleries_page_count_null ON galleries(gallery_id) WHERE page_count IS NULL AND files IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_galleries_created_at_unix_null ON galleries(gallery_id) WHERE (created_at_unix IS NULL OR created_at_unix = 0) AND created_at IS NOT NULL",
        ]
        for stmt in index_statements:
            await conn.execute(text(stmt))

        # --- 正規化タグテーブル ---
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gallery_tags (
                gallery_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (gallery_id, tag),
                FOREIGN KEY (gallery_id) REFERENCES galleries(gallery_id) ON DELETE CASCADE
            )
        """))
        
        # gallery_tags インデックス（バッチ実行）
        gallery_tags_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_gallery_tags_tag_gallery ON gallery_tags(tag, gallery_id)",
            "CREATE INDEX IF NOT EXISTS idx_gallery_tags_gallery_tag ON gallery_tags(gallery_id, tag)",
            "CREATE INDEX IF NOT EXISTS idx_gallery_tags_tag ON gallery_tags(tag)",
        ]
        for stmt in gallery_tags_indexes:
            await conn.execute(text(stmt))
        
        # galleriesテーブルの追加インデックス（page_countフィルタリング高速化用）
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_galleries_page_count ON galleries(page_count)"))
        # manga_type + page_count 複合インデックス（min_pages/max_pagesフィルター高速化用）
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_galleries_type_page_count ON galleries(manga_type, page_count)"))

        # --- FTS5 テーブル（存在しない場合のみ作成）---
        # 不要なトリガだけクリーンアップ（古いスキーマ互換用）
        await conn.execute(text("DROP TRIGGER IF EXISTS galleries_created_at_unix_ai"))
        await conn.execute(text("DROP TRIGGER IF EXISTS galleries_created_at_unix_au"))
        
        # FTS5テーブルが存在するかチェック
        fts_exists = await conn.execute(text(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='galleries_fts'"
        ))
        if not fts_exists.scalar():
            await conn.execute(text("""
                CREATE VIRTUAL TABLE galleries_fts USING fts5(
                    japanese_title,
                    tags,
                    characters,
                    content='galleries',
                    content_rowid='gallery_id',
                    tokenize = 'unicode61 remove_diacritics 2 tokenchars ''-_+&/#:."()[]{}'''
                )
            """))

        # --- FTS 同期トリガ（存在しない場合のみ作成）---
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS galleries_ai AFTER INSERT ON galleries BEGIN
                INSERT INTO galleries_fts(rowid, japanese_title, tags, characters)
                VALUES (new.gallery_id, new.japanese_title, new.tags, new.characters);
            END
        """))
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS galleries_ad AFTER DELETE ON galleries BEGIN
                INSERT INTO galleries_fts(galleries_fts, rowid, japanese_title, tags, characters)
                VALUES ('delete', old.gallery_id, old.japanese_title, old.tags, old.characters);
            END
        """))
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS galleries_au AFTER UPDATE OF japanese_title, tags, characters ON galleries BEGIN
                INSERT INTO galleries_fts(galleries_fts, rowid, japanese_title, tags, characters)
                VALUES ('delete', old.gallery_id, old.japanese_title, old.tags, old.characters);
                INSERT INTO galleries_fts(rowid, japanese_title, tags, characters)
                VALUES (new.gallery_id, new.japanese_title, new.tags, new.characters);
            END
        """))

        # --- gallery_tags 同期トリガ ---
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS gallery_tags_ai AFTER INSERT ON galleries BEGIN
                INSERT OR IGNORE INTO gallery_tags(gallery_id, tag)
                SELECT NEW.gallery_id, LOWER(TRIM(value))
                FROM json_each(CASE WHEN json_valid(NEW.tags) THEN NEW.tags ELSE '[]' END)
                WHERE value IS NOT NULL AND TRIM(value) <> '';
            END
        """))
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS gallery_tags_ad AFTER DELETE ON galleries BEGIN
                DELETE FROM gallery_tags WHERE gallery_id = OLD.gallery_id;
            END
        """))
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS gallery_tags_au AFTER UPDATE OF tags ON galleries BEGIN
                DELETE FROM gallery_tags WHERE gallery_id = NEW.gallery_id;
                INSERT OR IGNORE INTO gallery_tags(gallery_id, tag)
                SELECT NEW.gallery_id, LOWER(TRIM(value))
                FROM json_each(CASE WHEN json_valid(NEW.tags) THEN NEW.tags ELSE '[]' END)
                WHERE value IS NOT NULL AND TRIM(value) <> '';
            END
        """))

        # --- tag_stats テーブル ---
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tag_stats (
                tag TEXT PRIMARY KEY,
                count INTEGER NOT NULL DEFAULT 0
            )
        """))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tag_stats_count_tag ON tag_stats(count DESC, tag ASC)"))

        # --- tag_stats 同期トリガ ---
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS tag_stats_ins AFTER INSERT ON gallery_tags BEGIN
                INSERT INTO tag_stats(tag, count) VALUES (NEW.tag, 1)
                ON CONFLICT(tag) DO UPDATE SET count = count + 1;
            END
        """))
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS tag_stats_del AFTER DELETE ON gallery_tags BEGIN
                UPDATE tag_stats SET count = MAX(count - 1, 0) WHERE tag = OLD.tag;
            END
        """))

        # --- ランキングテーブル ---
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gallery_rankings (
                gallery_id INTEGER NOT NULL,
                ranking_type TEXT NOT NULL,
                score INTEGER NOT NULL DEFAULT 0,
                view_count INTEGER NOT NULL DEFAULT 0,
                last_updated TEXT,
                created_at TEXT,
                PRIMARY KEY (gallery_id, ranking_type),
                FOREIGN KEY (gallery_id) REFERENCES galleries(gallery_id) ON DELETE CASCADE
            )
        """))
        
        ranking_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_gallery_rankings_type_score ON gallery_rankings(ranking_type, score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_gallery_rankings_gallery_id ON gallery_rankings(gallery_id)",
            "CREATE INDEX IF NOT EXISTS idx_gallery_rankings_last_updated ON gallery_rankings(last_updated)",
        ]
        for stmt in ranking_indexes:
            await conn.execute(text(stmt))

        # --- tag_priorities テーブル ---
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tag_priorities (
                tag TEXT PRIMARY KEY,
                priority INTEGER NOT NULL DEFAULT 0
            )
        """))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tag_priorities_priority ON tag_priorities(priority)"))

    # --- 初期データ同期（別トランザクション） ---
    print("[init_db] 初期データ同期確認中...")
    async with engine.begin() as conn:
        # FTS REBUILD
        rebuild_result = await conn.execute(text("SELECT COUNT(*) = 0 FROM galleries_fts"))
        if rebuild_result.scalar():
            print("[init_db] FTS5 rebuild 開始...")
            rebuild_start = time.time()
            await conn.execute(text("INSERT INTO galleries_fts(galleries_fts) VALUES('rebuild')"))
            print(f"[init_db] FTS5 rebuild 完了 ({time.time() - rebuild_start:.1f}秒)")

        # gallery_tags 初期同期
        tag_sync_result = await conn.execute(text("SELECT COUNT(*) = 0 FROM gallery_tags"))
        if tag_sync_result.scalar():
            print("[init_db] gallery_tags 初期同期中...")
            sync_start = time.time()
            await conn.execute(text("""
                INSERT OR IGNORE INTO gallery_tags(gallery_id, tag)
                SELECT g.gallery_id, LOWER(TRIM(value))
                FROM galleries AS g,
                     json_each(CASE WHEN json_valid(g.tags) THEN g.tags ELSE '[]' END)
                WHERE value IS NOT NULL AND TRIM(value) <> ''
            """))
            print(f"[init_db] gallery_tags 同期完了 ({time.time() - sync_start:.1f}秒)")

        # tag_stats 初期バックフィル
        tag_stats_result = await conn.execute(text("SELECT COUNT(*) = 0 FROM tag_stats"))
        if tag_stats_result.scalar():
            print("[init_db] tag_stats バックフィル中...")
            stats_start = time.time()
            await conn.execute(text("""
                INSERT INTO tag_stats(tag, count)
                SELECT tag, COUNT(*) FROM gallery_tags GROUP BY tag
            """))
            print(f"[init_db] tag_stats 完了 ({time.time() - stats_start:.1f}秒)")

        # tag_priorities 初期同期
        print("[init_db] tag_priorities 同期中...")
        prio_start = time.time()
        try:
            translations_data = await _read_json_file(TAG_TRANSLATIONS_FILE, {})
            priorities = []
            if isinstance(translations_data, dict):
                for tag, data in translations_data.items():
                    if isinstance(data, dict):
                        p = data.get("priority")
                        if isinstance(p, int) and p != 0:
                            priorities.append({"tag": tag, "priority": p})
            
            if priorities:
                # 一旦クリア
                await conn.execute(text("DELETE FROM tag_priorities"))
                await conn.execute(
                    text("INSERT INTO tag_priorities(tag, priority) VALUES (:tag, :priority)"),
                    priorities
                )
            print(f"[init_db] tag_priorities 同期完了 ({len(priorities)}件, {time.time() - prio_start:.1f}秒)")
        except Exception as e:
            print(f"[init_db] tag_priorities 同期エラー: {e}")

        # 統計最適化
        await conn.execute(text("ANALYZE"))
        await conn.execute(text("PRAGMA optimize"))
    
    elapsed = time.time() - start_time
    print(f"[init_db] === データベース初期化完了 ({elapsed:.1f}秒) ===")

async def init_tracking_database():
    """
    tracking.db 初期化（シンプルな新スキーマ）。
    既存tracking.dbが古いスキーマの場合も create_all により必要テーブルのみ作成される前提。
    破壊的マイグレーションが必要になった場合は明示的にファイル削除する運用とする。
    """
    async with tracking_engine.begin() as conn:
        await conn.run_sync(TrackingBase.metadata.create_all)

    async with tracking_engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.execute(text("PRAGMA synchronous=NORMAL"))
        await conn.execute(text("PRAGMA cache_size=10000"))
        await conn.execute(text("PRAGMA temp_store=MEMORY"))
        await conn.execute(text("PRAGMA foreign_keys=ON"))

        # 新テーブル用インデックス（モデル定義のunique indexに加えて）
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_logs_last_viewed ON user_logs(last_viewed_at)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_impressions_last_shown ON impressions(last_shown_at)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tag_preferences_user ON tag_preferences(user_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_search_history_last ON search_history(last_searched_at)"))

# =========================
# FTS ユーティリティ
# =========================
@lru_cache(maxsize=2048)
def _escape_for_fts(s: Optional[str]) -> str:
    if not s:
        return ""
    return s.replace('"', '""')

@lru_cache(maxsize=512)
def _terms(col: str, value: str) -> str:
    raw = value.strip()
    if not raw:
        return f'{col} : ""'
    toks = [t for t in TOKEN_SPLIT_RE.split(raw) if t]
    if len(toks) >= 3:
        return " NEAR/8 ".join(f"{col} : {t}" for t in toks)
    if len(toks) == 2:
        return f"{col} : {toks[0]} NEAR/6 {col} : {toks[1]}"
    return f'{col} : "{_escape_for_fts(raw)}"'

@lru_cache(maxsize=512)
def _build_fts_query(title: Optional[str] = None, character: Optional[str] = None) -> str:
    clauses = []
    if title:
        clauses.append(f'japanese_title : "{_escape_for_fts(title)}"')
    if character:
        clauses.append(_terms("characters", character))
    return " AND ".join(clauses) if clauses else ""

_KNOWN_TAGS_CACHE: Set[str] = set()
_KNOWN_TAGS_FETCHED_AT: float = 0.0
_TAG_CACHE_LOCK = asyncio.Lock()


async def _get_known_tag_set(db_session: AsyncSession) -> Set[str]:
    global _KNOWN_TAGS_CACHE, _KNOWN_TAGS_FETCHED_AT
    
    # Fast path check without lock
    now = time.time()
    if _KNOWN_TAGS_CACHE and now - _KNOWN_TAGS_FETCHED_AT < 300:
        return _KNOWN_TAGS_CACHE

    async with _TAG_CACHE_LOCK:
        # Double-check inside lock
        now = time.time()
        if _KNOWN_TAGS_CACHE and now - _KNOWN_TAGS_FETCHED_AT < 300:
            return _KNOWN_TAGS_CACHE

        try:
            result = await db_session.execute(text("SELECT tag FROM tag_stats"))
            rows = result.fetchall()
        except Exception as exc:
            print(f"タグ一覧の取得に失敗しました: {exc}")
            return _KNOWN_TAGS_CACHE

        tags: Set[str] = set()
        for row in rows:
            tag_value = row[0] if isinstance(row, (list, tuple)) else row.tag if hasattr(row, "tag") else None
            if not tag_value:
                continue
            normalized = str(tag_value).strip().lower()
            if normalized:
                tags.add(normalized)

        _KNOWN_TAGS_CACHE = tags
        _KNOWN_TAGS_FETCHED_AT = now
        return _KNOWN_TAGS_CACHE


def _parse_tag_terms(tag_query: Optional[str], known_tags: Optional[Set[str]] = None) -> Tuple[str, ...]:
    if not tag_query:
        return tuple()
    raw = tag_query.strip()
    if not raw:
        return tuple()

    raw = raw.replace("\u3000", " ")
    separators = [",", ";", "\n"]
    separators_present = any(sep in raw for sep in separators)

    def _split(text_value: str) -> List[str]:
        try:
            return [token.strip() for token in shlex.split(text_value) if token.strip()]
        except ValueError:
            return [part.strip() for part in text_value.split() if part.strip()]

    candidates: List[str]
    if separators_present:
        normalized = raw
        for sep in separators:
            normalized = normalized.replace(sep, " ")
        candidates = _split(normalized)
    else:
        candidates = _split(raw)

    if not candidates:
        return tuple()

    lowered = [candidate.lower() for candidate in candidates]

    # 既知のタグを考慮した解析を常に実行（複数語のタグを正しく処理するため）
    if known_tags and len(lowered) > 1:
        resolved: List[str] = []
        idx = 0
        total = len(lowered)
        while idx < total:
            match = None
            # 最長一致で既知のタグを探す
            for end in range(total, idx, -1):
                candidate = " ".join(lowered[idx:end])
                if candidate in known_tags:
                    match = candidate
                    idx = end
                    break
            if match is None:
                match = lowered[idx]
                idx += 1
            resolved.append(match)
        lowered = resolved

    seen: Set[str] = set()
    unique_terms: List[str] = []
    for term in lowered:
        if term and term not in seen:
            unique_terms.append(term)
            seen.add(term)
    return tuple(unique_terms)

_GALLERY_FIELD_NAMES: Tuple[str, ...] = (
    "gallery_id",
    "japanese_title",
    "tags",
    "characters",
    "manga_type",
    "created_at",
    "page_count",
    "created_at_unix",
)

_FTS_CANDIDATE_MULTIPLIER = 5
_FTS_CANDIDATE_MAX = 1000
_MAX_GALLERY_ID_SENTINEL = 9_223_372_036_854_775_807

def _serialize_gallery(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    result = {field: mapping[field] for field in _GALLERY_FIELD_NAMES}
    if "files" in mapping:
        result["files"] = mapping["files"]
    return result

# =========================
# クエリ: 高速検索
# =========================

def _build_tag_exists_clause(alias: str, tag_terms: Tuple[str, ...]) -> Tuple[List[str], Dict[str, Any]]:
    """タグ AND 検索を GROUP BY/HAVING ではなく、EXISTS 連鎖で最適化。
    gallery_tags(tag, gallery_id) の複合インデックスを最大限活用する。
    """
    where_list: List[str] = []
    params: Dict[str, Any] = {}
    for idx, term in enumerate(tag_terms):
        key = f"tag_{idx}"
        where_list.append(
            f"EXISTS (SELECT 1 FROM gallery_tags AS gt INDEXED BY idx_gallery_tags_gallery_tag WHERE gt.gallery_id = {alias}.gallery_id AND gt.tag = :{key})"
        )
        params[key] = term
    return where_list, params


def _build_tag_not_exists_clause(alias: str, tag_terms: Tuple[str, ...]) -> Tuple[List[str], Dict[str, Any]]:
    """指定タグを含むギャラリーを除外する WHERE 句を生成する。"""
    where_list: List[str] = []
    params: Dict[str, Any] = {}
    for idx, term in enumerate(tag_terms):
        key = f"exclude_tag_{idx}"
        where_list.append(
            f"NOT EXISTS (SELECT 1 FROM gallery_tags AS gt INDEXED BY idx_gallery_tags_gallery_tag WHERE gt.gallery_id = {alias}.gallery_id AND gt.tag = :{key})"
        )
        params[key] = term
    return where_list, params

async def search_galleries_fast(
    db_session: AsyncSession,
    title: str = None,
    tag: str = None,
    character: str = None,
    q: str = None,
    limit: int = 50,
    offset: int = 0,
    after_created_at: str | None = None,
    after_gallery_id: int | None = None,
    exclude_tag: str | None = None,
    min_pages: int | None = None,
    max_pages: int | None = None,
    exclude_gallery_ids: Optional[Tuple[int, ...]] = None,
) -> List[Dict[str, Any]]:
    known_tags = await _get_known_tag_set(db_session)
    tag_terms = _parse_tag_terms(tag, known_tags)
    exclude_tag_terms = _parse_tag_terms(exclude_tag, known_tags)
    fts = _build_fts_query(title, character)

    async def run_query(use_fts: bool) -> List[Dict[str, Any]]:
        params: Dict[str, object] = {"limit": limit, "offset": offset}
        joins: List[str] = []
        where_clauses: List[str] = []
        cte_segment: Optional[str] = None

        # doujinshi & manga only
        where_clauses.append("g.manga_type IN ('doujinshi', 'manga')")

        if after_created_at is not None or after_gallery_id is not None:
            if after_created_at is not None:
                params["cursor_created_at"] = after_created_at
                params["cursor_gallery_id"] = (
                    after_gallery_id if after_gallery_id is not None else _MAX_GALLERY_ID_SENTINEL
                )
                where_clauses.append(
                    "(g.created_at < :cursor_created_at OR (g.created_at = :cursor_created_at AND g.gallery_id < :cursor_gallery_id))"
                )
            elif after_gallery_id is not None:
                params["cursor_gallery_id"] = after_gallery_id
                where_clauses.append("g.gallery_id < :cursor_gallery_id")

        if use_fts and fts:
            base_limit = max(limit, 1)
            prelimit = min(max(base_limit * _FTS_CANDIDATE_MULTIPLIER, base_limit), _FTS_CANDIDATE_MAX)
            # Offsetがある場合はprelimitを増やす必要があるかも知れないが、
            # FTSの場合は通常スコア順などでLimitをかけるため、Offsetとの相性は悪い。
            # ここでは単純にLimitをOffset分増やす簡易的な対応とするか、
            # あるいはFTS利用時はOffset非推奨とするか。
            # 一旦、prelimitにlimit + offsetを加味するようにする。
            if offset > 0:
                 prelimit = min(max((limit + offset) * _FTS_CANDIDATE_MULTIPLIER, (limit + offset)), _FTS_CANDIDATE_MAX)

            params["fts"] = fts
            params["prelimit"] = prelimit
            cte_segment = (
                "WITH fts_candidates AS (\n"
                "    SELECT rowid FROM galleries_fts WHERE galleries_fts MATCH :fts LIMIT :prelimit\n"
                ")"
            )
            joins.append("JOIN fts_candidates AS f ON f.rowid = g.gallery_id")
        else:
            # FTS unavailable: fall back to LIKE search
            if title:
                where_clauses.append("g.japanese_title LIKE :title_like")
                params["title_like"] = f"%{title}%"
            if character:
                where_clauses.append("g.characters LIKE :character_like")
                params["character_like"] = f"%{character}%"

        # Tag filters using EXISTS chain to enforce AND semantics
        if tag_terms:
            exists_clauses, exists_params = _build_tag_exists_clause("g", tag_terms)
            where_clauses.extend(exists_clauses)
            params.update(exists_params)

        if q:
            q_clauses = []
            q_clauses.append("g.japanese_title LIKE :q_like")
            q_clauses.append("g.characters LIKE :q_like")
            
            # タグとしてのチェック
            q_tag = q.lower()
            q_artist = f"artist:{q_tag}"
            
            # EXISTS句でのタグチェック
            q_clauses.append(f"EXISTS (SELECT 1 FROM gallery_tags WHERE gallery_id = g.gallery_id AND tag IN (:q_tag, :q_artist))")
            
            # 作品コードのチェック
            code_id = _extract_dlsite_id(q)
            if code_id:
                q_clauses.append("g.gallery_id = :code_id")
                params["code_id"] = code_id

            where_clauses.append("(" + " OR ".join(q_clauses) + ")")
            params["q_like"] = f"%{q}%"
            params["q_tag"] = q_tag
            params["q_artist"] = q_artist

        if exclude_tag_terms:
            not_exists_clauses, not_exists_params = _build_tag_not_exists_clause("g", exclude_tag_terms)
            where_clauses.extend(not_exists_clauses)
            params.update(not_exists_params)

        if exclude_gallery_ids:
            placeholders = []
            for idx, gallery_id in enumerate(exclude_gallery_ids):
                key = f"exclude_gallery_{idx}"
                placeholders.append(f":{key}")
                params[key] = gallery_id
            if placeholders:
                where_clauses.append(f"g.gallery_id NOT IN ({', '.join(placeholders)})")

        if min_pages is not None or max_pages is not None:
            min_val = max(min_pages or 0, 0)
            max_val = max_pages if max_pages is not None else 10_000
            if max_val < min_val:
                max_val = min_val
            params["min_pages"] = min_val
            params["max_pages"] = max_val
            where_clauses.append("g.page_count BETWEEN :min_pages AND :max_pages")

        sql_segments: List[str] = []
        if cte_segment:
            sql_segments.append(cte_segment)
        sql_segments.append("SELECT g.*")
        sql_segments.append("FROM galleries AS g")
        if joins:
            sql_segments.extend(joins)
        if where_clauses:
            sql_segments.append("WHERE " + " AND ".join(where_clauses))
        sql_segments.append("ORDER BY g.created_at DESC, g.gallery_id DESC")
        sql_segments.append("LIMIT :limit OFFSET :offset")
        sql = "\n".join(sql_segments)

        result = await db_session.execute(text(sql), params)
        return [_serialize_gallery(row) for row in result.mappings()]

    if fts:
        try:
            results = await run_query(True)
            if results:
                return results
        except Exception:
            pass
    try:
        return await run_query(False)
    except Exception:
        return await search_galleries(
            db_session,
            title=title,
            tag=tag,
            character=character,
            q=q,
            limit=limit,
            offset=offset,
            exclude_tag=exclude_tag,
        )
async def search_galleries(
    db_session: AsyncSession,
    title: str = None,
    tag: str = None,
    character: str = None,
    q: str = None,
    limit: int = None,
    offset: int = None,
    exclude_tag: str = None,
) -> List[Dict[str, Any]]:
    stmt = select(
        Gallery.gallery_id,
        Gallery.japanese_title,
        Gallery.tags,
        Gallery.characters,
        Gallery.files,
        Gallery.manga_type,
        Gallery.created_at,
        Gallery.page_count,
        Gallery.created_at_unix,
    ).where(Gallery.manga_type.in_(['doujinshi', 'manga']))

    if title:
        stmt = stmt.where(Gallery.japanese_title.like(f"%{title}%"))
    if tag:
        # JSON 文字列に対する LIKE は遅いが、フォールバックとして残す
        stmt = stmt.where(Gallery.tags.like(f'%"{tag}"%'))
    if exclude_tag:
        stmt = stmt.where(~Gallery.tags.like(f'%"{exclude_tag}"%'))
    if character:
        stmt = stmt.where(Gallery.characters.like(f'%"{character}"%'))

    stmt = stmt.order_by(Gallery.created_at.desc(), Gallery.gallery_id.desc())
    if offset:
        stmt = stmt.offset(offset)
    if limit:
        stmt = stmt.limit(limit)

    result = await db_session.execute(stmt)
    return [_serialize_gallery(row) for row in result.mappings()]


SESSION_TAG_LOOKBACK_DAYS = 30
SESSION_TAG_MAX_PAGE_VIEWS = 200
SESSION_TAG_MIN_RECENCY_WEIGHT = 0.2


def _extract_gallery_id_from_page_url(page_url: Optional[str]) -> Optional[int]:
    if not page_url:
        return None
    try:
        parsed = urlparse(page_url)
    except Exception:
        return None

    query = parse_qs(parsed.query)
    for key in ("id", "gallery_id"):
        values = query.get(key)
        if values:
            try:
                return int(values[0])
            except (TypeError, ValueError):
                continue

    path = parsed.path or ""
    match = re.search(r"/(?:viewer|gallery)/(\d+)", path)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _normalize_tag_value(tag_value: Any) -> Optional[str]:
    if isinstance(tag_value, str):
        normalized = tag_value.strip().lower()
        return normalized or None
    return None


# TODO: TrackingMangaView が未定義のため、この関数は実行時に NameError になる。
#       session_id, page_url, time_on_page 等のカラムも既存モデルに存在しない。
async def build_session_tag_profile(
    db_session: AsyncSession,
    session_id: str,
    lookback_days: int = SESSION_TAG_LOOKBACK_DAYS,
    max_page_views: int = SESSION_TAG_MAX_PAGE_VIEWS,
) -> Dict[str, float]:
    if not session_id:
        return {}

    try:
        async with get_tracking_db_session() as tracking_db:
            stmt = (
                select(TrackingMangaView)
                .where(TrackingMangaView.session_id == session_id)
                .order_by(TrackingMangaView.id.desc())
                .limit(max_page_views)
            )
            result = await tracking_db.execute(stmt)
            page_views = result.scalars().all()
    except Exception as exc:
        print(f"セッションプロファイル取得エラー: {exc}")
        return {}

    if not page_views:
        return {}

    now = datetime.now(timezone.utc)
    gallery_scores: Dict[int, float] = {}
    lookback_seconds = max(lookback_days, 1) * 86400

    for view in page_views:
        gallery_id = _extract_gallery_id_from_page_url(getattr(view, "page_url", None))
        if not gallery_id:
            continue

        duration_seconds: Optional[float] = None
        if isinstance(view.time_on_page, (int, float)):
            duration_seconds = max(float(view.time_on_page), 0.0)
        elif getattr(view, "view_start", None) and getattr(view, "view_end", None):
            try:
                start = datetime.fromisoformat(view.view_start)
                end = datetime.fromisoformat(view.view_end)
                # タイムゾーン情報がない場合はUTCと仮定
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                duration_seconds = max((end - start).total_seconds(), 0.0)
            except ValueError:
                duration_seconds = None

        duration_weight = 1.0
        if duration_seconds and duration_seconds > 0:
            duration_weight += min(duration_seconds / 120.0, 3.0)

        recency_weight = 1.0
        if getattr(view, "view_start", None):
            try:
                view_time = datetime.fromisoformat(view.view_start)
                # タイムゾーン情報がない場合はUTCと仮定
                if view_time.tzinfo is None:
                    view_time = view_time.replace(tzinfo=timezone.utc)
                age_seconds = max((now - view_time).total_seconds(), 0.0)
                if lookback_seconds > 0:
                    recency_weight = 1.0 - min(age_seconds / lookback_seconds, 1.0)
                    recency_weight = max(recency_weight, SESSION_TAG_MIN_RECENCY_WEIGHT)
            except ValueError:
                recency_weight = 1.0

        weight = duration_weight * recency_weight
        gallery_scores[gallery_id] = gallery_scores.get(gallery_id, 0.0) + weight

    if not gallery_scores:
        return {}

    gallery_ids = list(gallery_scores.keys())
    try:
        stmt = select(Gallery.gallery_id, Gallery.tags).where(Gallery.gallery_id.in_(gallery_ids))
        result = await db_session.execute(stmt)
        gallery_rows = result.all()
    except Exception as exc:
        print(f"ギャラリータグ取得エラー: {exc}")
        return {}

    tag_weights: Dict[str, float] = {}
    for row in gallery_rows:
        raw_tags = row.tags if hasattr(row, 'tags') else row[1]
        try:
            tags_data = json.loads(raw_tags) if isinstance(raw_tags, str) else raw_tags
        except (TypeError, json.JSONDecodeError):
            tags_data = []

        if not isinstance(tags_data, list):
            continue

        gallery_weight = gallery_scores.get(row[0] if isinstance(row, tuple) else row.gallery_id, 0.0)
        if gallery_weight <= 0:
            continue

        for tag_value in tags_data:
            normalized = _normalize_tag_value(tag_value)
            if not normalized:
                continue
            tag_weights[normalized] = tag_weights.get(normalized, 0.0) + gallery_weight

    if not tag_weights:
        return {}

    max_weight = max(tag_weights.values())
    if max_weight <= 0:
        return {}
    return {tag: weight / max_weight for tag, weight in tag_weights.items() if weight > 0}

RECOMMENDATION_CANDIDATE_MULTIPLIER = 10
RECOMMENDATION_CANDIDATE_MAX = 200

async def get_recommended_galleries(
    db_session: AsyncSession,
    gallery_id: Optional[int] = None,
    limit: int = 8,
    exclude_tag: Optional[str] = None,
    session_tag_weights: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    高速化されたおすすめギャラリー取得関数。
    - 重いJOINとGROUP BYを避け、軽量なクエリで関連候補を絞り込む。
    - 2段階のフェッチ戦略でDB負荷を軽減。
    """
    base_gallery_tags: Tuple[str, ...] = tuple()
    exclude_ids: Set[int] = set()

    if gallery_id is not None:
        exclude_ids.add(gallery_id)
        stmt = select(Gallery.tags).where(Gallery.gallery_id == gallery_id)
        result = await db_session.execute(stmt)
        gallery_tags_json = result.scalar_one_or_none()
        if gallery_tags_json:
            try:
                raw_tags = json.loads(gallery_tags_json)
                normalized = []
                for tag in raw_tags or []:
                    if isinstance(tag, str):
                        cleaned_tag = tag.strip().lower()
                        if cleaned_tag:
                            normalized.append(cleaned_tag)
                base_gallery_tags = tuple(normalized[:15])  # 参照するタグ数を制限
            except (TypeError, json.JSONDecodeError):
                pass

    known_tags = await _get_known_tag_set(db_session)
    exclude_terms = _parse_tag_terms(exclude_tag, known_tags)
    
    candidate_galleries: Dict[int, Dict[str, Any]] = {}

    # ステップ1: タグに基づいて関連性の高い候補ギャラリーIDを取得
    if base_gallery_tags:
        candidate_limit = min(limit * RECOMMENDATION_CANDIDATE_MULTIPLIER, RECOMMENDATION_CANDIDATE_MAX)
        params: Dict[str, Any] = {"limit": candidate_limit}
        
        tag_placeholders = []
        for i, tag in enumerate(base_gallery_tags):
            key = f"tag_{i}"
            params[key] = tag
            tag_placeholders.append(f":{key}")

        # gallery_tagsテーブルのみを使い、高速にマッチ数を集計
        candidate_query = f"""
            SELECT
                gt.gallery_id,
                COUNT(gt.gallery_id) as match_count
            FROM gallery_tags AS gt
            WHERE gt.tag IN ({', '.join(tag_placeholders)})
            GROUP BY gt.gallery_id
            ORDER BY match_count DESC
            LIMIT :limit
        """
        
        candidate_id_rows = await db_session.execute(text(candidate_query), params)
        candidate_ids_with_score = {row.gallery_id: row.match_count for row in candidate_id_rows}

        if candidate_ids_with_score:
            # 除外IDを削除
            for ex_id in exclude_ids:
                candidate_ids_with_score.pop(ex_id, None)

            # ステップ2: 候補IDのギャラリー情報を取得
            if candidate_ids_with_score:
                galleries_stmt = select(Gallery).where(Gallery.gallery_id.in_(candidate_ids_with_score.keys()))
                gallery_results = await db_session.execute(galleries_stmt)

                for gallery_row in gallery_results.scalars().all():
                    serialized = _serialize_gallery(gallery_row.__dict__)
                    serialized["_match_score"] = candidate_ids_with_score.get(gallery_row.gallery_id, 0)
                    candidate_galleries[gallery_row.gallery_id] = serialized

    # 取得したギャラリーをスコア順にソート
    sorted_galleries = sorted(candidate_galleries.values(), key=lambda g: -g.get("_match_score", 0))

    # exclude_tagでフィルタリング
    if exclude_terms:
        # このフィルタリングはPython側で行う方が効率的
        def check_exclude(g: Dict[str, Any]) -> bool:
            try:
                g_tags = {t.strip().lower() for t in json.loads(g.get("tags") or "[]")}
                return not any(ex_tag in g_tags for ex_tag in exclude_terms)
            except (TypeError, json.JSONDecodeError):
                return True
        sorted_galleries = [g for g in sorted_galleries if check_exclude(g)]

    galleries = sorted_galleries[:limit]
    final_gallery_ids = {g["gallery_id"] for g in galleries}
    exclude_ids.update(final_gallery_ids)

    # ステップ3: 結果が不足している場合、最新のギャラリーで補完
    if len(galleries) < limit:
        remaining = limit - len(galleries)
        fallback_params: Dict[str, Any] = {"limit": remaining}
        fallback_clauses = ["g.manga_type IN ('doujinshi', 'manga')"]

        if exclude_terms:
            not_exists, not_params = _build_tag_not_exists_clause("g", exclude_terms)
            fallback_clauses.extend(not_exists)
            fallback_params.update(not_params)

        # これまでに見つかった全てのIDを除外
        if exclude_ids:
            placeholders = []
            for i, ex_id in enumerate(exclude_ids):
                key = f"ex_id_{i}"
                fallback_params[key] = ex_id
                placeholders.append(f":{key}")
            fallback_clauses.append(f"g.gallery_id NOT IN ({','.join(placeholders)})")

        fallback_sql = text(
            "SELECT * FROM galleries AS g WHERE " + " AND ".join(fallback_clauses) + 
            " ORDER BY g.created_at DESC, g.gallery_id DESC LIMIT :limit"
        )
        
        fallback_result = await db_session.execute(fallback_sql, fallback_params)
        fallback_galleries = [_serialize_gallery(row) for row in fallback_result.mappings()]
        galleries.extend(fallback_galleries)

    # ステップ4: パーソナライズスコアで最終的な並び替え
    if session_tag_weights and galleries:
        normalized_weights = {
            key.strip().lower(): value
            for key, value in session_tag_weights.items()
            if isinstance(key, str) and key.strip()
        }
        if normalized_weights:
            for idx, item in enumerate(galleries):
                item["_base_order"] = idx
                score = 0.0
                try:
                    tags_list = json.loads(item.get("tags") or "[]")
                    for tag_value in tags_list:
                        normalized = _normalize_tag_value(tag_value)
                        if normalized:
                            score += normalized_weights.get(normalized, 0.0)
                except (TypeError, json.JSONDecodeError):
                    pass
                item["_personal_score"] = score
            
            # 安定ソート: パーソナルスコアが同じ場合は元の順序を維持
            galleries.sort(key=lambda item: (-item.get("_personal_score", 0.0), item.get("_base_order", 0)))
            
            for item in galleries:
                score = item.pop("_personal_score", None)
                item.pop("_base_order", None)
                item.pop("_match_score", None) # 不要なキーを削除
                if score and score > 0:
                    item["personal_score"] = round(float(score), 4)

    return galleries[:limit]

def _derive_filename(url: str) -> str:
    trimmed = url.split("?", 1)[0].rstrip("/")
    candidate = trimmed.split("/")[-1] if trimmed else ""
    return candidate or "download.bin"


_IMAGE_RESOLVER_FAILURE_AT: float = 0.0
_IMAGE_RESOLVER_FAILURE_COOLDOWN = 120.0
_IMAGE_RESOLVER_TIMEOUT = 15.0  # lib側のフェッチタイムアウト(10秒)より長く設定
_IMAGE_RESOLVER_READY = False
_IMAGE_RESOLVER_READY_UNTIL: float = 0.0
_IMAGE_RESOLVER_READY_TTL = 3600.0  # 1時間キャッシュ（lib側のrefresh_intervalと同期）
_IMAGE_RESOLVER_LOCK = asyncio.Lock()


async def _ensure_image_resolver_ready() -> bool:
    """Initialise the image resolver without blocking the event loop."""

    global _IMAGE_RESOLVER_FAILURE_AT, _IMAGE_RESOLVER_READY, _IMAGE_RESOLVER_READY_UNTIL

    now = time.monotonic()
    
    # 高速パス: 既に準備完了している場合
    if _IMAGE_RESOLVER_READY and now < _IMAGE_RESOLVER_READY_UNTIL:
        return True
    
    # 失敗クールダウン中
    if _IMAGE_RESOLVER_FAILURE_AT and now - _IMAGE_RESOLVER_FAILURE_AT < _IMAGE_RESOLVER_FAILURE_COOLDOWN:
        return False

    async with _IMAGE_RESOLVER_LOCK:
        # ロック内で再チェック
        now = time.monotonic()
        if _IMAGE_RESOLVER_READY and now < _IMAGE_RESOLVER_READY_UNTIL:
            return True
        if _IMAGE_RESOLVER_FAILURE_AT and now - _IMAGE_RESOLVER_FAILURE_AT < _IMAGE_RESOLVER_FAILURE_COOLDOWN:
            return False

        try:
            await asyncio.wait_for(ImageUriResolver.async_synchronize(), timeout=_IMAGE_RESOLVER_TIMEOUT)
            _IMAGE_RESOLVER_READY = True
            _IMAGE_RESOLVER_READY_UNTIL = now + _IMAGE_RESOLVER_READY_TTL
            return True
        except asyncio.TimeoutError:
            print("ImageUriResolver 同期がタイムアウトしました")
        except Exception as exc:
            print(f"ImageUriResolver 初期化エラー: {exc}")

        _IMAGE_RESOLVER_FAILURE_AT = now
        _IMAGE_RESOLVER_READY = False
        return False


async def geturl(gi: Dict[str, Any]) -> List[str]:
    """filesからhashのリストを返す（ImageUriResolverは/proxy/で適用）"""
    files = gi.get("files", []) or []
    if not files:
        return []

    # hashだけを返す（拡張子はavif決め打ち）
    hashes: List[str] = []
    for f in files:
        h = (f.get("hash") or "").lower()
        if h:
            hashes.append(h)
    return hashes

# =========================
# リクエスト/レスポンスモデル
# =========================
class DownloadRequest(BaseModel):
    urls: List[str]

class DownloadResponse(BaseModel):
    results: List[Dict[str, str]]

class MultipartDownloadRequest(BaseModel):
    url: str
    chunk_size: int = 1024 * 1024
    max_connections: int = 4
    headers: Optional[Dict[str, str]] = None
    as_attachment: bool = False
    filename: Optional[str] = None

class TagTranslationsUpdateRequest(BaseModel):
    translations: Dict[str, Any] = Field(default_factory=dict)
    base_version: Optional[str] = None
    message: Optional[str] = None
    auto_save: bool = False


class TagTranslationsRollbackRequest(BaseModel):
    version: str


class TagCategoryModel(BaseModel):
    id: str
    label: str
    tags: List[str] = Field(default_factory=list)


class TagCategoriesUpdateRequest(BaseModel):
    categories: List[TagCategoryModel] = Field(default_factory=list)


class SnapshotCreateRequest(BaseModel):
    history: List[Dict[str, Any]] = Field(default_factory=list)
    hidden_tags: List[str] = Field(default_factory=list)
    likes: List[str] = Field(default_factory=list)
    tag_usage: Dict[str, int] = Field(default_factory=dict)


class SnapshotDataResponse(BaseModel):
    history: List[Dict[str, Any]] = Field(default_factory=list)
    hidden_tags: List[str] = Field(default_factory=list)
    likes: List[str] = Field(default_factory=list)
    tag_usage: Dict[str, int] = Field(default_factory=dict)
    expires_at: Optional[str] = None

class SearchRequest(BaseModel):
    title: Optional[str] = None
    tag: Optional[str] = None
    exclude_tag: Optional[str] = None
    character: Optional[str] = None
    limit: int = 50
    after_created_at: Optional[str] = None
    after_gallery_id: Optional[int] = None
    min_pages: Optional[int] = None
    max_pages: Optional[int] = None

# ---- 新ログ API用モデル ----

class ViewLogRequest(BaseModel):
    """閲覧ログ記録リクエスト"""
    user_id: str
    manga_id: int
    duration: int = 0  # 今回の閲覧時間（秒）
    max_page: int = 0  # 今回の最大閲覧ページ
    page_count: Optional[int] = None  # 作品の総ページ数

class ImpressionRequest(BaseModel):
    """インプレッション記録リクエスト"""
    user_id: str
    manga_ids: List[int] = Field(default_factory=list)  # 表示された漫画IDリスト
    tags: List[str] = Field(default_factory=list)  # 表示された漫画のタグリスト

class ClickRequest(BaseModel):
    """クリック記録リクエスト"""
    user_id: str
    manga_id: int
    tags: List[str] = Field(default_factory=list)  # クリックされた漫画のタグ

class SearchLogRequest(BaseModel):
    """検索タグ記録リクエスト"""
    user_id: str
    tags: List[str] = Field(default_factory=list)

# =========================
# ルータ
# =========================
@app.get("/", response_class=HTMLResponse)
async def read_index():
    return _serve_cached_html("template/index.html")

@app.get("/viewer", response_class=HTMLResponse)
async def read_viewer():
    return _serve_cached_html("template/viewer.html")

@app.get("/history", response_class=HTMLResponse)
async def read_history():
    return _serve_cached_html("template/history.html")


@app.get("/tag-editor", response_class=HTMLResponse)
async def read_tag_editor():
    return _serve_cached_html("template/tag-editor.html")

@app.get("/api/recommendations")
async def api_recommendations(
    gallery_id: Optional[int] = None,
    limit: int = 8,
    exclude_tag: Optional[str] = None,
    session_id: Optional[str] = None,
):
    try:
        async with get_db_session() as db:
            session_tag_weights: Optional[Dict[str, float]] = None
            if session_id:
                try:
                    # NOTE: さらなる高速化のため、この結果をキャッシュすることも検討できます
                    session_tag_weights = await build_session_tag_profile(db, session_id)
                except Exception as exc:
                    print(f"おすすめ個人化プロファイル作成エラー: {exc}")
                    session_tag_weights = None

            # 1. 高速化された関数でギャラリーリストを取得
            results = await get_recommended_galleries(
                db,
                gallery_id=gallery_id,
                limit=limit,
                exclude_tag=exclude_tag,
                session_tag_weights=session_tag_weights,
            )

            if not results:
                return {"results": [], "count": 0}

            # 2. geturlの呼び出しを並列化
            geturl_tasks = []
            files_data_cache = []  # パース結果をキャッシュ
            
            for result in results:
                try:
                    files_data = json.loads(result.get("files")) if isinstance(result.get("files"), str) else result.get("files")
                except (json.JSONDecodeError, TypeError):
                    files_data = []
                files_list = files_data if isinstance(files_data, list) else []
                files_data_cache.append(files_list)  # キャッシュに保存
                gallery_info = {"gallery_id": result["gallery_id"], "files": files_list}
                geturl_tasks.append(geturl(gallery_info))
            
            # asyncio.gatherで全てのgeturlコルーチンを並列実行
            image_urls_results = await asyncio.gather(*geturl_tasks, return_exceptions=True)

            # 3. 結果をマージして最終的なペイロードを生成
            payload: List[Dict[str, Any]] = []
            for i, result in enumerate(results):
                image_urls = image_urls_results[i]
                if isinstance(image_urls, Exception):
                    print(f"geturlがギャラリーID {result['gallery_id']} で失敗: {image_urls}")
                    image_urls = []

                # キャッシュからfiles_listを取得
                files_list = files_data_cache[i]
                
                page_count = result.get("page_count")
                if not isinstance(page_count, int) or page_count < 0:
                    page_count = len(files_list)

                # レスポンスから 'files' キーを削除
                final_result = {k: v for k, v in result.items() if k != "files"}
                final_result["image_urls"] = image_urls
                final_result["page_count"] = page_count
                payload.append(final_result)

            return {"results": payload, "count": len(payload)}

    except Exception as e:
        print("おすすめ取得APIで予期せぬエラーが発生しました")
        raise HTTPException(status_code=500, detail=f"おすすめ取得エラー: {str(e)}")

@app.get("/search")
async def search_galleries_get(
    title: Optional[str] = None,
    tag: Optional[str] = None,
    q: Optional[str] = None,
    exclude_tag: Optional[str] = None,
    character: Optional[str] = None,
    limit: int = 50,
    after_created_at: Optional[str] = None,
    after_gallery_id: Optional[int] = None,
    min_pages: Optional[int] = None,
    max_pages: Optional[int] = None,
    sort_by: Optional[str] = None,
    offset: int = 0,
    page: int = 1,
):
    MAX_LIMIT = 200
    if limit > MAX_LIMIT:
        limit = MAX_LIMIT

    try:
        # q にカンマが含まれている場合はタグ検索として扱う
        if q and "," in q:
            if tag:
                tag = f"{tag},{q}"
            else:
                tag = q
            q = None

        # q が URL の場合はプロダクトコード（RJ****等）に変換する
        if q:
            extracted_code = _extract_dlsite_product_code(q)
            if extracted_code:
                q = extracted_code

        # page パラメータが指定されている場合は offset を上書き
        if page > 1:
            offset = (page - 1) * limit

        # タイトル検索、自由入力検索（作品コード、作者名等含む）、またはタグによる作者検索の場合は、
        # ランキングソートを無視して新着順（デフォルト）を優先する
        if (title or q or (tag and "artist:" in tag.lower())) and sort_by in ['daily', 'weekly', 'monthly', 'yearly', 'all_time']:
            sort_by = None

        # キャッシュキーを生成
        cache_key = _make_search_results_cache_key(
            title=title,
            tag=tag,
            q=q,
            exclude_tag=exclude_tag,
            character=character,
            limit=limit,
            offset=offset,
            after_created_at=after_created_at,
            after_gallery_id=after_gallery_id,
            min_pages=min_pages,
            max_pages=max_pages,
            sort_by=sort_by,
        )

        # キャッシュをチェック
        cached = _get_cached_search_results(cache_key)
        if cached is not None:
            cached_results, cached_total_count = cached
            
            # キャッシュからレスポンスを構築
            if sort_by and sort_by in ['daily', 'weekly', 'monthly', 'yearly', 'all_time']:
                total_pages = (cached_total_count + limit - 1) // limit if cached_total_count > 0 else 1
                return {
                    "results": cached_results,
                    "count": len(cached_results),
                    "total": cached_total_count,
                    "total_count": cached_total_count,
                    "total_pages": total_pages,
                    "has_more": (offset + limit) < cached_total_count,
                    "ranking_type": sort_by,
                    "cached": True,
                }
            else:
                next_after_created_at = None
                next_after_gallery_id = None
                if cached_results and len(cached_results) == limit:
                    last_item = cached_results[-1]
                    next_after_created_at = last_item.get("created_at")
                    next_after_gallery_id = last_item.get("gallery_id")
                
                total_pages = (cached_total_count + limit - 1) // limit if cached_total_count > 0 else 1
                is_cursor_pagination = after_created_at is not None or after_gallery_id is not None
                if is_cursor_pagination:
                    has_more = len(cached_results) == limit
                else:
                    has_more = (offset + limit) < cached_total_count

                return {
                    "results": cached_results,
                    "count": len(cached_results),
                    "total": cached_total_count,
                    "total_count": cached_total_count,
                    "total_pages": total_pages,
                    "has_more": has_more,
                    "next_after_created_at": next_after_created_at,
                    "next_after_gallery_id": next_after_gallery_id,
                    "cached": True,
                }

        async with get_db_session() as db:
                # known_tagsを一度だけ取得してキャッシュ（複数回のawaitを避ける）
                known_tags = await _get_known_tag_set(db)
                
                # sort_byパラメータがランキングの場合は特別処理
                if sort_by and sort_by in ['daily', 'weekly', 'monthly', 'yearly', 'all_time']:
                    # ランキングファイルからIDを読み込む
                    ranking_ids = await _load_ranking_ids(sort_by)
                    
                    # タグ検索の場合はタグでフィルタリング
                    if tag:
                        tag_terms = _parse_tag_terms(tag, known_tags)
                       
                        if tag_terms:
                            # タグを持つギャラリーIDを取得
                            tag_exists_clauses, tag_params = _build_tag_exists_clause("g", tag_terms)
                            
                            tag_query = f"""
                                SELECT DISTINCT g.gallery_id
                                FROM galleries AS g
                                WHERE {' AND '.join(tag_exists_clauses)}
                            """
                            
                            tag_result = await db.execute(text(tag_query), tag_params)
                            tag_gallery_ids = {row.gallery_id for row in tag_result.fetchall()}
                            
                            # ランキングIDとタグを持つギャラリーIDの共通部分を取得
                            filtered_ranking_ids = [gid for gid in ranking_ids if gid in tag_gallery_ids]
                        else:
                            filtered_ranking_ids = []
                    else:
                        filtered_ranking_ids = ranking_ids
                    
                    # ページ数フィルターがある場合は、キャッシュ付きでフィルタリング
                    if filtered_ranking_ids and (min_pages is not None or max_pages is not None):
                        min_val = max(min_pages or 0, 0)
                        max_val = max_pages if max_pages is not None else 10_000
                        if max_val < min_val:
                            max_val = min_val
                        
                        # ページ数範囲に該当するギャラリーIDのセットを取得（キャッシュ付き）
                        valid_ids_set = await _get_page_range_gallery_ids(min_val, max_val)
                        
                        # ランキング順序を維持したまま、有効なIDだけを残す
                        filtered_ranking_ids = [gid for gid in filtered_ranking_ids if gid in valid_ids_set]
                    
                    # ランキング順にギャラリー情報を取得
                    if filtered_ranking_ids:
                        # リミットとオフセットを適用
                        start_idx = offset
                        end_idx = min(start_idx + limit, len(filtered_ranking_ids))
                        paginated_ids = filtered_ranking_ids[start_idx:end_idx]
                        
                        placeholders = ', '.join([f':id_{i}' for i in range(len(paginated_ids))])
                        params = {f'id_{i}': gallery_id for i, gallery_id in enumerate(paginated_ids)}
    
                        order_case_parts = [f"WHEN :id_{i} THEN {i}" for i in range(len(paginated_ids))]
                        order_case = "CASE g.gallery_id " + " ".join(order_case_parts) + f" ELSE {len(paginated_ids)} END"
    
                        query = f"""
                            SELECT
                                g.gallery_id,
                                g.japanese_title,
                                g.tags,
                                g.characters,
                                g.files,
                                g.manga_type,
                                g.created_at,
                                g.page_count,
                                g.created_at_unix
                            FROM galleries AS g
                            WHERE g.gallery_id IN ({placeholders})
                            ORDER BY {order_case}
                        """
                        
                        result = await db.execute(text(query), params)
                        results = [_serialize_gallery(row) for row in result.mappings()]
                        
                        # ファイル情報の処理を並列化
                        await _process_results_with_image_urls(results)
                    else:
                        results = []
                else:
                    # 通常の検索 - 検索クエリとCOUNTクエリを並列実行
                    search_task = search_galleries_fast(
                        db,
                        title=title,
                        tag=tag,
                        q=q,
                        exclude_tag=exclude_tag,
                        character=character,
                        limit=limit,
                        offset=offset,
                        after_created_at=after_created_at,
                        after_gallery_id=after_gallery_id,
                        min_pages=min_pages,
                        max_pages=max_pages,
                    )
                    
                    count_task = _get_search_count(
                        title=title,
                        tag=tag,
                        exclude_tag=exclude_tag,
                        character=character,
                        q=q,
                        min_pages=min_pages,
                        max_pages=max_pages,
                        known_tags=known_tags,
                    )
                    
                    # 検索とCOUNTを同時に実行
                    results, total_count = await asyncio.gather(search_task, count_task)
                         
                    # ファイル情報の処理を並列化
                    await _process_results_with_image_urls(results)

                # レスポンス形式を統一
                if sort_by and sort_by in ['daily', 'weekly', 'monthly', 'yearly', 'all_time']:
                    # ランキングの場合のレスポンス
                    # filtered_ranking_idsには既にタグフィルターとページ数フィルターが適用済み
                    total_count = len(filtered_ranking_ids)
                    total_pages = (total_count + limit - 1) // limit if total_count > 0 else 1
                    
                    # キャッシュに保存
                    _set_cached_search_results(cache_key, results, total_count)
                    
                    return {
                        "results": results,
                        "count": len(results),
                        "total": total_count,
                        "total_count": total_count,
                        "total_pages": total_pages,
                        "has_more": (offset + limit) < total_count,
                        "ranking_type": sort_by,
                    }
                else:
                    # 通常検索の場合のレスポンス
                    next_after_created_at = None
                    next_after_gallery_id = None
                    if results and len(results) == limit:
                        last_item = results[-1]
                        next_after_created_at = last_item.get("created_at")
                        next_after_gallery_id = last_item.get("gallery_id")
                    
                    total_pages = (total_count + limit - 1) // limit if total_count > 0 else 1
                    
                    # カーソルページネーションを使っているかどうかで判定ロジックを変える
                    # after_created_at/id が指定されている場合は offset ベースの判定はできない（offset=0前提のため）
                    is_cursor_pagination = after_created_at is not None or after_gallery_id is not None
                    if is_cursor_pagination:
                        has_more = len(results) == limit
                    else:
                        has_more = (offset + limit) < total_count

                    # キャッシュに保存
                    _set_cached_search_results(cache_key, results, total_count)

                    return {
                        "results": results,
                        "count": len(results),
                        "total": total_count,
                        "total_count": total_count,
                        "total_pages": total_pages,
                        "has_more": has_more,
                        "next_after_created_at": next_after_created_at,
                        "next_after_gallery_id": next_after_gallery_id,
                    }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"検索エラー: {str(e)}")

@app.get("/proxy/{hash_or_path:path}")
async def proxy_request(
    hash_or_path: str,
    thumbnail: bool = False,
    small: bool = False,
):
    """
    高速化された画像プロキシエンドポイント
    - hashが渡された場合: ImageUriResolverでURLを解決してからプロキシ
    - URLが渡された場合: 従来通りそのままプロキシ（後方互換性）
    - ストリーミングレスポンス: TTFBを最小化
    - インメモリキャッシュ: 小さい画像を即座に返却
    - 並列Range Request: 大きい画像を高速ダウンロード

    クエリパラメータ:
    - thumbnail: サムネイル画像を取得する場合はtrue
    - small: 小さいサムネイルを取得する場合はtrue（thumbnailがtrueの時のみ有効）
    """
    # URLかhashかを判定
    if hash_or_path.startswith(("http://", "https://")) or "gold-usergeneratedcontent.net" in hash_or_path:
        url = hash_or_path if hash_or_path.startswith(("http://", "https://")) else f"https://{hash_or_path}"
    else:
        resolver_ready = await _ensure_image_resolver_ready()
        if not resolver_ready:
            raise HTTPException(status_code=503, detail="ImageUriResolver is not ready")
        
        try:
            image = SimpleNamespace(
                hash=hash_or_path.lower(),
                has_avif=True,
                has_webp=True,
                has_jxl=False,
            )
            resolved = ImageUriResolver.get_image_uri(
                image,
                "avif",
                is_thumbnail=thumbnail,
                is_small=small,
            )
            url = f"https://{resolved}" if not resolved.startswith("http") else resolved
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"URL resolution failed: {str(e)}")
    
    if "gold-usergeneratedcontent.net" not in url:
        raise HTTPException(status_code=400, detail="Invalid URL")

    # キャッシュキーの生成（URLをベースにする）
    cache_key = f"{url}"
    # サムネイルまたはスモール画像の場合はキャッシュをチェック
    use_cache = thumbnail or small
    if use_cache:
        cached = thumbnail_cache.get(cache_key)
        if cached:
            data, content_type = cached
            return Response(
                content=data,
                media_type=content_type,
                headers={'Cache-Control': 'public, max-age=3600', 'Access-Control-Allow-Origin': '*', 'X-Cache': 'HIT'},
            )
    
    # === キャッシュチェック（最速パス）===
    cached = await _image_proxy_cache.get(url)
    if cached:
        data, content_type = cached
        return Response(
            content=data,
            media_type=content_type,
            headers={
                'Cache-Control': 'public, max-age=86400',  # 24時間
                'Access-Control-Allow-Origin': '*',
                'X-Cache': 'HIT',
            },
        )
    
    headers = _build_headers()
    
    # セッション取得（グローバル優先）
    session = global_session
    created_local = False
    if session is None:
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=20, ttl_dns_cache=300)
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60, connect=10, sock_read=30),
            headers=headers,
            connector=connector,
        )
        created_local = True
    
    max_retries = 2
    retry_delay = 0.5
    
    # 定数（ループ外で定義してパフォーマンス向上）
    PARALLEL_THRESHOLD = 256 * 1024  # 256KB以上で並列化
    CHUNK_SIZE = 128 * 1024  # 128KBチャンク
    MAX_PARALLEL_CONNECTIONS = 6
    STREAMING_THRESHOLD = 512 * 1024  # 512KB以下はバッファ
    
    try:
        for attempt in range(max_retries + 1):
            try:
                # === GETリクエスト開始（HEADリクエスト省略で1往復節約）===
                async with session.get(url, headers=headers) as resp:
                    if resp.status >= 500:
                        if attempt < max_retries:
                            await asyncio.sleep(retry_delay * (2 ** attempt))
                            continue
                        raise HTTPException(status_code=resp.status, detail=f"Upstream 5xx: {resp.status}")
                    
                    if resp.status >= 400:
                        raise HTTPException(status_code=resp.status, detail=f"Upstream error: {resp.status}")
                    
                    content_type = resp.content_type or "application/octet-stream"
                    resp_length = resp.headers.get("Content-Length")
                    total_length = int(resp_length) if resp_length and resp_length.isdigit() else None
                    accept_ranges = resp.headers.get("Accept-Ranges", "")
                    supports_range = accept_ranges == "bytes"
                    
                    # === 大きいファイル + Range対応 → 並列ダウンロードに切り替え ===
                    if content_type.startswith("image/") and supports_range and total_length and total_length > PARALLEL_THRESHOLD:
                        # 最初のチャンクを読み込み済みの場合があるので、respを閉じて並列ダウンロードに切り替え
                        # ここではrespはまだデータを読んでいないので、単に並列ダウンロードに移行
                        pass  # async withを抜けた後に並列ダウンロードを実行
                    else:
                        # === 通常ダウンロード（ストリーミング対応）===
                        if content_type.startswith("image/"):
                            # 小さいファイル: 一括読み込み + キャッシュ
                            if total_length and total_length <= STREAMING_THRESHOLD:
                                data = await resp.read()
                                await _image_proxy_cache.set(url, data, content_type)
                                if use_cache:
                                    thumbnail_cache.put(cache_key, data, content_type)
                                return Response(
                                    content=data,
                                    media_type=content_type,
                                    headers={
                                        'Cache-Control': 'public, max-age=86400',
                                        'Access-Control-Allow-Origin': '*',
                                        'X-Cache': 'MISS',
                                        'X-Download-Mode': 'buffered',
                                    },
                                )
                            
                            # 中〜大ファイル（Range非対応）: ストリーミングレスポンス
                            async def stream_content():
                                async for chunk in resp.content.iter_chunked(64 * 1024):
                                    yield chunk
                            
                            response_headers = {
                                'Cache-Control': 'public, max-age=86400',
                                'Access-Control-Allow-Origin': '*',
                                'X-Cache': 'MISS',
                                'X-Download-Mode': 'streaming',
                            }
                            if total_length:
                                response_headers['Content-Length'] = str(total_length)
                            
                            return StreamingResponse(
                                stream_content(),
                                media_type=content_type,
                                headers=response_headers,
                            )
                        else:
                            # 非画像コンテンツ
                            content = await resp.text()
                            return HTMLResponse(content=content)
                
                # === 並列ダウンロード（async with外で実行）===
                # ここに到達 = 大きいファイル + Range対応
                try:
                    data, final_content_type = await _download_with_ranges(
                        session, url, headers,
                        chunk_size=CHUNK_SIZE,
                        max_connections=MAX_PARALLEL_CONNECTIONS,
                        total_length=total_length,
                        content_type=content_type,
                    )
                    result_content_type = final_content_type or content_type or "image/avif"
                    
                    # キャッシュに保存
                    await _image_proxy_cache.set(url, data, result_content_type)
                    if use_cache:
                        thumbnail_cache.put(cache_key, data, result_content_type)
                    
                    return Response(
                        content=data,
                        media_type=result_content_type,
                        headers={
                            'Cache-Control': 'public, max-age=86400',
                            'Access-Control-Allow-Origin': '*',
                            'X-Cache': 'MISS',
                            'X-Download-Mode': 'parallel',
                        },
                    )
                except MultipartDownloadError:
                    # 並列ダウンロード失敗 → 通常のGETで再取得
                    async with session.get(url, headers=headers) as fallback_resp:
                        if fallback_resp.status >= 400:
                            raise HTTPException(status_code=fallback_resp.status, detail=f"Upstream error: {fallback_resp.status}")
                        data = await fallback_resp.read()
                        ct = fallback_resp.content_type or "image/avif"
                        await _image_proxy_cache.set(url, data, ct)
                        if use_cache:
                            thumbnail_cache.put(cache_key, data, ct)
                        return Response(
                            content=data,
                            media_type=ct,
                            headers={
                                'Cache-Control': 'public, max-age=86400',
                                'Access-Control-Allow-Origin': '*',
                                'X-Cache': 'MISS',
                                'X-Download-Mode': 'fallback',
                            },
                        )
                        
            except aiohttp.ClientError as e:
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                raise HTTPException(status_code=500, detail=f"Proxy client error: {str(e)}")
    finally:
        if created_local:
            await session.close()


# キャッシュ統計API
@app.get("/api/proxy-cache-stats")
async def get_proxy_cache_stats():
    """プロキシキャッシュの統計情報を取得"""
    return _image_proxy_cache.stats()


async def _download_entire(session: aiohttp.ClientSession, url: str, headers: Dict[str, str]) -> Tuple[bytes, Optional[str]]:
    async with session.get(url, headers=headers) as resp:
        if resp.status >= 400:
            raise HTTPException(status_code=resp.status, detail=f"upstream responded with status {resp.status}")
        content_type = resp.headers.get("Content-Type")
        data = await resp.read()
        return data, content_type

async def _download_with_ranges(
    session: aiohttp.ClientSession,
    url: str,
    headers: Dict[str, str],
    chunk_size: int,
    max_connections: int,
    total_length: int,
    content_type: Optional[str],
) -> Tuple[bytes, Optional[str]]:
    if total_length <= 0:
        raise MultipartDownloadError("Invalid total length")

    first_end = min(chunk_size, total_length) - 1
    range_headers = headers.copy()
    range_headers["Range"] = f"bytes=0-{first_end}"

    async with session.get(url, headers=range_headers) as resp:
        if resp.status == 206:
            content_type = content_type or resp.headers.get("Content-Type")
            first_chunk = await resp.read()
            if len(first_chunk) != first_end + 1:
                raise MultipartDownloadError("First chunk size mismatch")
        elif resp.status == 200:
            data = await resp.read()
            content_type = content_type or resp.headers.get("Content-Type")
            return data, content_type
        else:
            raise MultipartDownloadError(f"Range request failed: status={resp.status}")

    ranges: List[Tuple[int, int]] = []
    offset = first_end + 1
    while offset < total_length:
        end = min(offset + chunk_size, total_length) - 1
        ranges.append((offset, end))
        offset = end + 1

    sem = asyncio.Semaphore(max_connections)

    async def fetch_range(start: int, end: int) -> Tuple[int, bytes]:
        chunk_headers = headers.copy()
        chunk_headers["Range"] = f"bytes={start}-{end}"
        async with sem:
            async with session.get(url, headers=chunk_headers) as resp:
                if resp.status != 206:
                    raise MultipartDownloadError(f"Range request failed: status={resp.status}")
                data = await resp.read()
                expected = end - start + 1
                if len(data) != expected:
                    raise MultipartDownloadError("Chunk size mismatch")
                return start, data

    tasks = [asyncio.create_task(fetch_range(start, end)) for start, end in ranges]
    try:
        other_chunks: List[Tuple[int, bytes]] = []
        if tasks:
            other_chunks = await asyncio.gather(*tasks)
    except Exception:
        for task in tasks:
            task.cancel()
        raise

    parts = [(0, first_chunk)]
    parts.extend(other_chunks)
    parts.sort(key=lambda item: item[0])

    combined = b"".join(part for _, part in parts)
    return combined, content_type

async def _warmup_connections(session: aiohttp.ClientSession):
    """
    事前接続ウォームアップ
    - 複数ドメインに並列で接続してTCP/TLSハンドシェイクを事前完了
    - 接続プールを温めて初回リクエストを高速化
    """
    domains = [
        "a1.gold-usergeneratedcontent.net",
        "a2.gold-usergeneratedcontent.net",
        "a3.gold-usergeneratedcontent.net",
        "b.gold-usergeneratedcontent.net",
    ]
    
    async def warmup_single(domain: str) -> None:
        url = f"https://{domain}"
        try:
            async with session.head(url) as resp:
                print(f"事前接続成功: {domain} (Status: {resp.status})")
        except Exception as e:
            print(f"事前接続失敗: {domain} (Error: {str(e)[:50]})")
    
    # 並列でウォームアップ
    tasks = [warmup_single(domain) for domain in domains]
    await asyncio.gather(*tasks, return_exceptions=True)
    print(f"接続ウォームアップ完了 ({len(domains)}ドメイン)")


async def _download_single_url(session: aiohttp.ClientSession, url: str, headers: Dict[str, str]) -> Dict[str, str]:
    try:
        async with session.get(url, headers=headers) as resp:
            content = await resp.text()
            return {"url": url, "status": "success", "content": content, "status_code": str(resp.status)}
    except Exception as exc:
        return {"url": url, "status": "error", "content": str(exc), "status_code": "500"}

# 同時ダウンロード数の制限
MAX_CONCURRENT_DOWNLOADS = 20

@app.post("/download-multiple", response_model=DownloadResponse)
async def download_multiple(request: DownloadRequest):
    headers = _build_headers()
    
    # global_sessionがNoneの場合はローカルセッションを作成
    if global_session is None:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), headers=headers)
        created_local = True
    else:
        session = global_session
        created_local = False
    
    try:
        # セマフォで同時実行数を制限
        sem = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        
        async def _download_with_semaphore(url: str):
            async with sem:
                return await _download_single_url(session, url, headers)
        
        tasks = [_download_with_semaphore(url) for url in request.urls]
        results = await asyncio.gather(*tasks)
        return DownloadResponse(results=results)
    finally:
        if created_local:
            await session.close()

@app.post("/download-multipart")
async def download_multipart(request: MultipartDownloadRequest):
    if request.chunk_size <= 0:
        raise HTTPException(status_code=400, detail="chunk_size must be > 0")
    if request.chunk_size > 10 * 1024 * 1024:  # 10MB limit
         request.chunk_size = 10 * 1024 * 1024

    if request.max_connections <= 0:
        raise HTTPException(status_code=400, detail="max_connections must be > 0")
    if request.max_connections > 16:
        request.max_connections = 16

    headers = _build_headers(request.headers)
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        total_length: Optional[int] = None
        content_type: Optional[str] = None
        try:
            async with session.head(request.url, headers=headers) as head_resp:
                if head_resp.status < 400:
                    content_type = head_resp.headers.get("Content-Type")
                    length_header = head_resp.headers.get("Content-Length")
                    if length_header and length_header.isdigit():
                        total_length = int(length_header)
        except aiohttp.ClientError:
            pass

        use_multipart = total_length is not None and total_length > request.chunk_size
        if use_multipart:
            try:
                data, final_type = await _download_with_ranges(
                    session,
                    request.url,
                    headers,
                    request.chunk_size,
                    request.max_connections,
                    total_length,
                    content_type,
                )
            except MultipartDownloadError:
                data, final_type = await _download_entire(session, request.url, headers)
        else:
            data, final_type = await _download_entire(session, request.url, headers)

    response_headers: Dict[str, str] = {}
    if request.as_attachment:
        filename = request.filename or _derive_filename(request.url)
        response_headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    return Response(content=data, media_type=final_type or "application/octet-stream", headers=response_headers)

@app.get("/gallery/{gallery_id}")
async def get_gallery(gallery_id: int):
    try:
        async with get_db_session() as db:
            stmt = select(Gallery).where(Gallery.gallery_id == gallery_id)
            result = await db.execute(stmt)
            gallery = result.scalars().first()
            if not gallery:
                raise HTTPException(status_code=404, detail="ギャラリーが見つかりません")
            try:
                files_data = json.loads(gallery.files) if isinstance(gallery.files, str) else gallery.files
                files_list = files_data if isinstance(files_data, list) else []
                gallery_info = {"gallery_id": gallery.gallery_id, "files": files_list}
                image_urls = await geturl(gallery_info)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"files 解析エラー: {e}, gallery_id: {gallery.gallery_id}")
                image_urls = []
                files_list = []
            return {
                "gallery_id": gallery.gallery_id,
                "japanese_title": gallery.japanese_title,
                "artists": gallery.artists,
                "tags": gallery.tags,
                "characters": gallery.characters,
                "image_urls": image_urls,
                "manga_type": gallery.manga_type,
                "created_at": gallery.created_at,
                "page_count": gallery.page_count if isinstance(gallery.page_count, int) and gallery.page_count >= 0 else len(files_list),
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ギャラリー情報の取得エラー: {str(e)}")


@app.get("/api/artist/{artist_name}/works")
async def get_artist_works(
    artist_name: str,
    limit: int = 10,
    exclude_gallery_id: Optional[int] = None,
):
    """
    指定されたアーティストの他の作品を取得する。
    artist_nameには「artist:」プレフィックスなしのアーティスト名を渡す。
    """
    try:
        async with get_db_session() as db:
            # artist:プレフィックス付きのタグで検索
            artist_tag = f"artist:{artist_name}"
            
            # gallery_tagsテーブルを使ってアーティストの作品を検索
            query = """
                SELECT g.gallery_id, g.japanese_title, g.tags, g.characters, 
                       g.manga_type, g.created_at, g.page_count, g.files
                FROM galleries g
                INNER JOIN gallery_tags gt ON g.gallery_id = gt.gallery_id
                WHERE gt.tag = :artist_tag
                AND g.manga_type IN ('doujinshi', 'manga')
            """
            params: Dict[str, Any] = {"artist_tag": artist_tag}
            
            if exclude_gallery_id is not None:
                query += " AND g.gallery_id != :exclude_id"
                params["exclude_id"] = exclude_gallery_id
            
            query += " ORDER BY g.created_at_unix DESC, g.gallery_id DESC LIMIT :limit"
            params["limit"] = limit
            
            result = await db.execute(text(query), params)
            rows = result.fetchall()
            
            if not rows:
                return {"artist": artist_name, "results": [], "total": 0}
            
            # 結果を辞書形式に変換
            results = []
            for row in rows:
                gallery_dict = {
                    "gallery_id": row.gallery_id,
                    "japanese_title": row.japanese_title,
                    "tags": row.tags,
                    "characters": row.characters,
                    "manga_type": row.manga_type,
                    "created_at": row.created_at,
                    "page_count": row.page_count,
                    "files": row.files,
                }
                results.append(gallery_dict)
            
            # 画像URLを追加
            await _process_results_with_image_urls(results)
            
            return {
                "artist": artist_name,
                "results": results,
                "total": len(results),
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"アーティスト作品の取得エラー: {str(e)}")


# -------------------------
# タグ API（tag_stats 使用で高速化）
# -------------------------
@app.get("/api/tags")
async def get_tags(limit: int = 100, offset: int = 0, search: Optional[str] = None, user_id: Optional[str] = None):
    """
    tag-translations.json を用いて:
      - 日本語訳やエイリアスから英語タグへ正規化して検索可能にする
      - 英語タグ指定時も従来通り部分一致検索
    """
    try:

        search_counts: Dict[str, int] = {}
        if user_id:
            try:
                async with get_tracking_db_session() as tracking_db:
                    search_counts = await _get_user_search_counts(tracking_db, user_id)
            except Exception as e:
                print(f"Tracking DB error in get_tags: {e}")

        async with get_db_session() as db:
            params: Dict[str, Any] = {"limit": limit, "offset": offset}

            # ベースクエリと WHERE 条件を動的に組み立てる
            where_clauses = []
            search_param: Optional[str] = None

            if search:
                raw = search.strip()
                if raw:
                    # tag-translations.json を読み込み、日本語訳・エイリアス -> 英語タグ の逆引きマップを構築
                    try:
                        translations_data = await _read_json_file(TAG_TRANSLATIONS_FILE, {})
                    except Exception:
                        translations_data = {}

                    reverse_map: Dict[str, str] = {}

                    if isinstance(translations_data, Mapping):
                        for eng_tag, entry in translations_data.items():
                            if not eng_tag:
                                continue
                            eng_tag_str = str(eng_tag).strip()
                            if not eng_tag_str:
                                continue

                            # entry は string or {translation, aliases, ...}
                            if isinstance(entry, Mapping):
                                # translation
                                t = str(entry.get("translation") or "").strip().lower()
                                if t and t not in reverse_map:
                                    reverse_map[t] = eng_tag_str

                                # aliases
                                aliases = entry.get("aliases")
                                if isinstance(aliases, list):
                                    for alias in aliases:
                                        a = str(alias or "").strip().lower()
                                        if a and a not in reverse_map:
                                            reverse_map[a] = eng_tag_str
                            else:
                                # 素の文字列の場合も一応対応
                                t = str(entry or "").strip().lower()
                                if t and t not in reverse_map:
                                    reverse_map[t] = eng_tag_str

                    # 入力値を正規化してマッピング
                    key = raw.lower()
                    mapped = reverse_map.get(key)

                    if mapped:
                        # 日本語訳/エイリアスに完全一致した場合は、その英語タグに対して部分一致検索
                        search_param = f"%{mapped.lower()}%"
                    else:
                        # 一致しない場合は従来通り、入力値自体で部分一致 (英語タグ直接入力など)
                        search_param = f"%{raw.lower()}%"

                    where_clauses.append("LOWER(tag) LIKE :search")
                    params["search"] = search_param

            base_query = """
                SELECT ts.tag, ts.count, COALESCE(tp.priority, 0) as priority,
                0 as usage_count -- Placeholder, updated dynamically if search_counts exists
                FROM tag_stats ts
                LEFT JOIN tag_priorities tp ON ts.tag = tp.tag
            """
            base_total = """
                SELECT COUNT(*)
                FROM tag_stats ts
                LEFT JOIN tag_priorities tp ON ts.tag = tp.tag
            """

            # ユーザー検索履歴に基づく `usage_count` (CASE文生成)
            orders = []
            
            # 使用回数があるタグのためのソートロジック
            # 優先度ソート:
            # 1. Server-side Priority >= 1
            # 2. Dynamic Score (Count * Usage) if Usage > 0
            # 3. Else Count * small_factor (Default behavior)
            
            # CASE文で usage_count を定義
            # "CASE tag WHEN 't1' THEN c1 WHEN 't2' THEN c2 ... ELSE 0 END"
            usage_case_parts = []
            usage_params = {}
            
            if search_counts:
                # パラメータ数上限を考慮しつつ、上位のタグのみをSQLに埋め込む
                # SQLiteのパラメータ上限は通常999だが、安全のため上位200件程度にする
                sorted_usage = sorted(search_counts.items(), key=lambda x: x[1], reverse=True)[:200]
                
                for idx, (t_name, t_count) in enumerate(sorted_usage):
                    p_tag = f"u_tag_{idx}"
                    p_val = f"u_val_{idx}"
                    usage_case_parts.append(f"WHEN :u_tag_{idx} THEN :u_val_{idx}")
                    usage_params[p_tag] = t_name
                    usage_params[p_val] = t_count
                
                params.update(usage_params)

            if usage_case_parts:
                usage_sql = "CASE ts.tag " + " ".join(usage_case_parts) + " ELSE 0 END"
                # SELECT句を書き換えるのは面倒なので、Orderingで計算する
                # ただし、レスポンスに usage_count を含めるため、SELECT句も調整したいが
                # ここでは簡易的に、Orderingで制御し、usage_countは0のままでもフロントエンドでlocalStorageとマージできる
                # (フロントエンドは既にlocalStorageの値を使っているため)
                # しかし、Userは "Backend knows usage" と言っているため、backend dataを返すべきか。
                # ユーザーの要望「上に優先的に表示」はソートで達成できる。
                
                # スコア計算式:
                # IF priority >= 1 THEN (Priority * 10^9) -- Push to absolute top
                # ELSE IF usage > 0 THEN (Count * Usage)
                # ELSE Count * 0.00001 (Unused tags)
                
                # Priority >= 1 logic is handled by first order term.
                # Here we define the score for the rest.
                
                # Note: ts.count is integer.
                
                dynamic_score_sql = f"""
                    CASE
                        WHEN ({usage_sql}) > 0 THEN CAST(ts.count AS REAL) * ({usage_sql})
                        ELSE CAST(ts.count AS REAL) * 0.00001
                    END
                """
                
                orders = [
                    "CASE WHEN COALESCE(tp.priority, 0) >= 1 THEN 0 ELSE 1 END ASC",
                    "CASE WHEN COALESCE(tp.priority, 0) >= 1 THEN COALESCE(tp.priority, 0) ELSE 0 END DESC",
                    f"{dynamic_score_sql} DESC",
                    "ts.tag ASC"
                ]
            else:
                 orders = [
                    "CASE WHEN COALESCE(tp.priority, 0) >= 1 THEN 0 ELSE 1 END ASC",
                    "CASE WHEN COALESCE(tp.priority, 0) >= 1 THEN COALESCE(tp.priority, 0) ELSE 0 END DESC",
                    """CASE
                        WHEN COALESCE(tp.priority, 0) < 0 THEN CAST(ts.count AS REAL) / ABS(COALESCE(tp.priority, 0))
                        ELSE ts.count
                    END DESC""",
                    "ts.tag ASC"
                ]


            if where_clauses:
                where_sql = " WHERE " + " AND ".join(where_clauses)
                query = (
                    base_query
                    + where_sql
                    + " ORDER BY " + ", ".join(orders)
                    + " LIMIT :limit OFFSET :offset"
                )
                total_sql = base_total + where_sql
            else:
                query = (
                    base_query
                    + " ORDER BY " + ", ".join(orders)
                    + " LIMIT :limit OFFSET :offset"
                )
                total_sql = base_total

            total_result = await db.execute(text(total_sql), params)
            total_count = total_result.scalar()
            rows_result = await db.execute(text(query), params)
            rows = rows_result.fetchall()
            tags = []
            for row in rows:
                t_tag = row.tag
                t_count = row.count
                t_priority = row.priority
                t_usage = search_counts.get(t_tag.lower(), 0) if search_counts else 0
                
                tags.append({
                    "tag": t_tag,
                    "count": t_count,
                    "priority": t_priority,
                    "usage_count": t_usage
                })

            return {
                "tags": tags,
                "total": total_count,
                "has_more": (offset + limit) < (total_count or 0),
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"タグ情報の取得エラー: {str(e)}")

@app.get("/api/popular-tags")
async def get_popular_tags(limit: int = 10, offset: int = 0, exclude_existing: bool = True):
    """
    sa.dbから使用頻度の高いタグを取得するAPI
    既存の翻訳データにないタグのみを返すオプション付き
    """
    try:
        async with get_db_session() as db:
            # 基本クエリ：タグを使用頻度順に取得
            query = "SELECT tag, count FROM tag_stats ORDER BY count DESC, tag ASC LIMIT :limit OFFSET :offset"
            params: Dict[str, Any] = {"limit": limit, "offset": offset}
            
            # 既存の翻訳を除外する場合
            if exclude_existing:
                # 翻訳データを読み込み
                translations_data = await _read_json_file(TAG_TRANSLATIONS_FILE, {})
                existing_tags = set(translations_data.keys()) if translations_data else set()
                
                if existing_tags:
                    # 既存タグを除外するクエリを構築
                    placeholders = ', '.join([f':exclude_{i}' for i in range(len(existing_tags))])
                    for i, tag in enumerate(existing_tags):
                        params[f'exclude_{i}'] = tag
                    
                    query = f"""
                        SELECT tag, count FROM tag_stats
                        WHERE tag NOT IN ({placeholders})
                        ORDER BY count DESC, tag ASC
                        LIMIT :limit OFFSET :offset
                    """
            
            result = await db.execute(text(query), params)
            rows = result.fetchall()
            tags = [{"tag": r.tag, "count": r.count} for r in rows]
            return {"tags": tags, "count": len(tags)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"人気タグの取得エラー: {str(e)}")


@app.get("/tags", response_class=HTMLResponse)
async def read_tags():
    return _serve_cached_html("template/tags.html")


@app.get("/api/tag-translations")
async def get_tag_translations() -> Dict[str, Any]:
    async with tag_translations_lock:
        raw_data = await _read_json_file(TAG_TRANSLATIONS_FILE, {})
        if not isinstance(raw_data, Mapping):
            raise HTTPException(status_code=500, detail="タグ翻訳データの形式が正しくありません")

        storage: Dict[str, Dict[str, Any]] = {}
        response: Dict[str, Dict[str, Any]] = {}

        for raw_key, raw_value in raw_data.items():
            tag = str(raw_key or "").strip()
            if not tag:
                continue
            normalized = _normalize_translation_entry(raw_value)
            record: Dict[str, Any] = {"translation": normalized["translation"]}
            if normalized["description"]:
                record["description"] = normalized["description"]
            if normalized["aliases"]:
                record["aliases"] = normalized["aliases"]
            if normalized["priority"] != 0:
                record["priority"] = normalized["priority"]
            storage[tag] = record
            response[tag] = {
                "translation": normalized["translation"],
                "description": normalized["description"],
                "aliases": normalized["aliases"],
                "priority": normalized["priority"],
            }

        if raw_data != storage:
            await _write_json_file(TAG_TRANSLATIONS_FILE, storage, sort_keys=True)

        version = await _ensure_tag_translation_history_initialized(storage)
        if version:
            await tag_translations_state.set_version(version)
        else:
            version = await tag_translations_state.get_version()

    return {"translations": response, "version": version}


@app.post("/api/tag-translations")
async def update_tag_translations(request: TagTranslationsUpdateRequest) -> Dict[str, Any]:
    async with tag_translations_lock:
        current_raw = await _read_json_file(TAG_TRANSLATIONS_FILE, {})
        if not isinstance(current_raw, Mapping):
            raise HTTPException(status_code=500, detail="タグ翻訳データの形式が正しくありません")

        current_storage: Dict[str, Dict[str, Any]] = {}
        current_response: Dict[str, Dict[str, Any]] = {}

        for raw_key, raw_value in current_raw.items():
            tag = str(raw_key or "").strip()
            if not tag:
                continue
            normalized = _normalize_translation_entry(raw_value)
            record: Dict[str, Any] = {"translation": normalized["translation"]}
            if normalized["description"]:
                record["description"] = normalized["description"]
            if normalized["aliases"]:
                record["aliases"] = normalized["aliases"]
            if normalized["priority"] != 0:
                record["priority"] = normalized["priority"]
            current_storage[tag] = record
            current_response[tag] = normalized

        if current_storage != current_raw:
            await _write_json_file(TAG_TRANSLATIONS_FILE, current_storage, sort_keys=True)

        current_version = await _load_current_tag_translation_version()
        if current_version is None:
            current_version = await _ensure_tag_translation_history_initialized(current_storage)
        if current_version:
            await tag_translations_state.set_version(current_version)

        if request.base_version and current_version and request.base_version != current_version:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": "タグ翻訳データが他のユーザーによって更新されました。",
                    "version": current_version,
                    "translations": current_response,
                },
            )

        processed: Dict[str, Dict[str, Any]] = {}
        response_payload: Dict[str, Dict[str, Any]] = {}
        seen_tags: Set[str] = set()
        alias_map: Dict[str, str] = {}

        for raw_key, raw_value in request.translations.items():
            tag = str(raw_key or "").strip()
            if not tag:
                continue
            normalised_tag = _normalise_tag(tag)
            if normalised_tag in seen_tags:
                raise HTTPException(status_code=400, detail=f"タグが重複しています: {tag}")
            seen_tags.add(normalised_tag)

            normalized = _normalize_translation_entry(raw_value)
            record: Dict[str, Any] = {"translation": normalized["translation"]}
            if normalized["description"]:
                record["description"] = normalized["description"]
            if normalized["aliases"]:
                record["aliases"] = normalized["aliases"]
                for alias in normalized["aliases"]:
                    alias_norm = _normalise_tag(alias)
                    if not alias_norm:
                        continue
                    owner = alias_map.get(alias_norm)
                    if owner and owner != tag:
                        raise HTTPException(status_code=400, detail=f"検索キーワードが重複しています: {alias}")
                    alias_map[alias_norm] = tag

            if normalized["priority"] != 0:
                record["priority"] = normalized["priority"]

            processed[tag] = record
            response_payload[tag] = normalized

        await _write_json_file(TAG_TRANSLATIONS_FILE, processed, sort_keys=True)

        # tag_priorities DB同期
        try:
            priorities_to_insert = []
            for tag, entry in processed.items():
                p = entry.get("priority", 0)
                if isinstance(p, int) and p != 0:
                    priorities_to_insert.append({"tag": tag, "priority": p})
            
            async with get_db_session() as db:
                async with db.begin():
                    await db.execute(text("DELETE FROM tag_priorities"))
                    if priorities_to_insert:
                        await db.execute(
                            text("INSERT INTO tag_priorities(tag, priority) VALUES (:tag, :priority)"),
                            priorities_to_insert
                        )
        except Exception as e:
            print(f"tag_priorities sync error: {e}")

        reason = (request.message or "").strip()
        if reason:
            reason = reason[:200]
        else:
            reason = "auto-save" if request.auto_save else "manual"

        entry = await _record_tag_translation_version(
            processed,
            reason=reason,
            auto=request.auto_save,
            parent_version=current_version,
        )
        new_version = entry.get("version")

    return {
        "status": "ok",
        "count": len(processed),
        "version": new_version,
        "translations": response_payload,
    }


@app.get("/api/tag-translations/versions")
async def get_tag_translation_versions() -> Dict[str, Any]:
    async with tag_translations_lock:
        versions = await _load_tag_translation_versions()
    versions_sorted = sorted(
        versions,
        key=lambda item: item.get("created_at", "") or item["version"],
        reverse=True,
    )
    return {"versions": versions_sorted}


@app.get("/api/tag-translations/updates")
async def wait_for_tag_translation_updates(
    since: Optional[str] = Query(default=None),
    timeout: float = Query(default=25.0, ge=5.0, le=120.0),
) -> Dict[str, Any]:
    version, changed = await tag_translations_state.wait_for_update(since, timeout)
    return {"version": version, "changed": changed}


@app.post("/api/tag-translations/rollback")
async def rollback_tag_translations(request: TagTranslationsRollbackRequest) -> Dict[str, Any]:
    async with tag_translations_lock:
        versions = await _load_tag_translation_versions()
        version_ids = {entry.get("version") for entry in versions}
        if request.version not in version_ids:
            raise HTTPException(status_code=404, detail="指定されたバージョンが見つかりません")

        snapshot_path = TAG_TRANSLATIONS_HISTORY_DIR / f"{request.version}.json"
        if not snapshot_path.exists():
            raise HTTPException(status_code=404, detail="スナップショットが見つかりません")

        snapshot_raw = await _read_json_file(snapshot_path, {})
        if not isinstance(snapshot_raw, Mapping):
            raise HTTPException(status_code=500, detail="スナップショットの形式が正しくありません")

        storage: Dict[str, Dict[str, Any]] = {}
        response_payload: Dict[str, Dict[str, Any]] = {}
        for raw_key, raw_value in snapshot_raw.items():
            tag = str(raw_key or "").strip()
            if not tag:
                continue
            normalized = _normalize_translation_entry(raw_value)
            record: Dict[str, Any] = {"translation": normalized["translation"]}
            if normalized["description"]:
                record["description"] = normalized["description"]
            if normalized["aliases"]:
                record["aliases"] = normalized["aliases"]
            storage[tag] = record
            response_payload[tag] = normalized

        await _write_json_file(TAG_TRANSLATIONS_FILE, storage, sort_keys=True)

        current_version = await _load_current_tag_translation_version()
        entry = await _record_tag_translation_version(
            storage,
            reason="rollback",
            auto=False,
            parent_version=current_version,
            restored_from=request.version,
        )
        new_version = entry.get("version")

    return {
        "status": "ok",
        "version": new_version,
        "restored_from": request.version,
        "translations": response_payload,
    }


@app.get("/api/tag-categories")
async def get_tag_categories() -> Dict[str, Any]:
    data = await _read_json_file(TAG_CATEGORIES_FILE, [])
    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail="タグカテゴリデータの形式が正しくありません")

    categories: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        raw_tags = item.get("tags", [])
        if isinstance(raw_tags, list):
            tags = [tag for tag in raw_tags if isinstance(tag, str)]
        else:
            tags = []
        categories.append(
            {
                "id": str(item.get("id", "")),
                "label": str(item.get("label", "")),
                "tags": tags,
            }
        )

    return {"categories": categories}


@app.post("/api/tag-categories")
async def update_tag_categories(request: TagCategoriesUpdateRequest) -> Dict[str, Any]:
    processed: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    for category in request.categories:
        cat_id = category.id.strip()
        if not cat_id:
            raise HTTPException(status_code=400, detail="カテゴリIDを入力してください")
        normalised_id = cat_id.lower()
        if normalised_id in seen_ids:
            raise HTTPException(status_code=400, detail=f"カテゴリIDが重複しています: {cat_id}")
        seen_ids.add(normalised_id)

        label = category.label.strip() or cat_id
        deduped_tags: List[str] = []
        seen_tags: Set[str] = set()
        for raw_tag in category.tags:
            tag = (raw_tag or "").strip()
            if not tag:
                continue
            normalised_tag = tag.lower()
            if normalised_tag in seen_tags:
                continue
            seen_tags.add(normalised_tag)
            deduped_tags.append(tag)

        processed.append({"id": cat_id, "label": label, "tags": deduped_tags})

    await _write_json_file(TAG_CATEGORIES_FILE, processed)
    return {"status": "ok", "count": len(processed)}

# =========================
# ユーザーデータ引き継ぎ
# =========================


async def _store_snapshot(db: AsyncSession, payload: Dict[str, Any], *, code: Optional[str] = None) -> UserSnapshot:
    now = datetime.utcnow()
    created_at = now.isoformat()
    expires_at = (now + timedelta(days=SNAPSHOT_EXPIRY_DAYS)).isoformat()
    payload_json = json.dumps(payload, ensure_ascii=False)

    snapshot_code = _normalise_tag(code or "")
    if snapshot_code:
        stmt = select(UserSnapshot).where(UserSnapshot.code == snapshot_code)
        result = await db.execute(stmt)
        existing = result.scalars().first()
    else:
        existing = None

    if existing:
        existing.payload = payload_json
        existing.last_accessed = created_at
        existing.expires_at = expires_at
        await db.commit()
        await db.refresh(existing)
        return existing

    for _ in range(5):
        snapshot_code = _generate_snapshot_code()
        stmt = select(UserSnapshot).where(UserSnapshot.code == snapshot_code)
        result = await db.execute(stmt)
        if result.scalar() is None:
            break
    else:
        raise HTTPException(status_code=500, detail="引き継ぎコードの生成に失敗しました")

    snapshot = UserSnapshot(
        code=snapshot_code,
        payload=payload_json,
        created_at=created_at,
        expires_at=expires_at,
        last_accessed=created_at,
    )
    db.add(snapshot)
    await db.commit()
    await db.refresh(snapshot)
    return snapshot


@app.post("/api/history/snapshots")
async def create_history_snapshot(request: SnapshotCreateRequest, http_request: Request) -> Dict[str, Any]:
    payload = {
        "version": 1,
        "history": _sanitize_snapshot_history(request.history),
        "hidden_tags": _sanitize_string_list(request.hidden_tags),
        "likes": _sanitize_string_list(request.likes),
        "tag_usage": _sanitize_tag_usage(request.tag_usage),
    }

    try:
        async with get_tracking_db_session() as db:
            snapshot = await _store_snapshot(db, payload)
            base_url = str(http_request.base_url).rstrip('/')
            restore_url = f"{base_url}/history?transfer={snapshot.code}"
            return {
                "status": "ok",
                "code": snapshot.code,
                "restore_url": restore_url,
                "expires_at": snapshot.expires_at,
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"スナップショットの保存に失敗しました: {exc}") from exc


@app.get("/api/history/snapshots/{code}")
async def load_history_snapshot(code: str) -> SnapshotDataResponse:
    normalised = _normalise_tag(code)
    if not normalised:
        raise HTTPException(status_code=400, detail="コードが不正です")

    try:
        async with get_tracking_db_session() as db:
            stmt = select(UserSnapshot).where(UserSnapshot.code == normalised)
            result = await db.execute(stmt)
            snapshot = result.scalars().first()
            if not snapshot:
                raise HTTPException(status_code=404, detail="スナップショットが見つかりません")

            try:
                payload = json.loads(snapshot.payload)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=500, detail="スナップショットデータが破損しています") from exc

            expires_at = snapshot.expires_at
            if expires_at:
                try:
                    expiry = datetime.fromisoformat(expires_at)
                    if expiry < datetime.utcnow():
                        raise HTTPException(status_code=410, detail="スナップショットの有効期限が切れています")
                except ValueError:
                    pass

            snapshot.last_accessed = datetime.utcnow().isoformat()
            await db.commit()

            data = SnapshotDataResponse(
                history=_sanitize_snapshot_history(payload.get("history", [])),
                hidden_tags=_sanitize_string_list(payload.get("hidden_tags", [])),
                likes=_sanitize_string_list(payload.get("likes", [])),
                tag_usage=_sanitize_tag_usage(payload.get("tag_usage", {})),
                expires_at=expires_at,
            )
            return data
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"スナップショットの読み込みに失敗しました: {exc}") from exc

# =========================
# 新ログAPI
# =========================

def _now_iso() -> str:
    return datetime.utcnow().isoformat()

@app.post("/api/logs/view")
async def api_log_view(request: ViewLogRequest):
    """
    閲覧ログ記録。
    同一user_id + manga_idの組み合わせでupsert（累積更新）。
    """
    try:
        async with get_tracking_db_session() as db:
            now = _now_iso()
            stmt = select(UserLog).where(
                UserLog.user_id == request.user_id,
                UserLog.manga_id == request.manga_id
            )
            result = await db.execute(stmt)
            log = result.scalars().first()
            
            if log:
                # 累積更新
                log.visit_count = (log.visit_count or 0) + 1
                log.total_duration = (log.total_duration or 0) + request.duration
                log.read_pages = max(log.read_pages or 0, request.max_page)
                # 平均時間を再計算（加重平均）
                if log.read_pages and log.read_pages > 0:
                    log.avg_time = log.total_duration // log.read_pages
                log.last_viewed_at = now
            else:
                # 新規作成
                avg = request.duration // request.max_page if request.max_page > 0 else 0
                log = UserLog(
                    user_id=request.user_id,
                    manga_id=request.manga_id,
                    avg_time=avg,
                    read_pages=request.max_page,
                    visit_count=1,
                    total_duration=request.duration,
                    last_viewed_at=now
                )
                db.add(log)
            
            await db.commit()
            return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"閲覧ログエラー: {e}")

@app.post("/api/logs/impression")
async def api_log_impression(request: ImpressionRequest):
    """
    インプレッション記録。
    表示された漫画IDとタグを記録。
    """
    try:
        async with get_tracking_db_session() as db:
            now = _now_iso()
            
            # 漫画ごとのインプレッション更新
            for manga_id in request.manga_ids:
                stmt = select(Impression).where(
                    Impression.user_id == request.user_id,
                    Impression.manga_id == manga_id
                )
                result = await db.execute(stmt)
                imp = result.scalars().first()
                
                if imp:
                    imp.shown_count = (imp.shown_count or 0) + 1
                    imp.last_shown_at = now
                else:
                    imp = Impression(
                        user_id=request.user_id,
                        manga_id=manga_id,
                        shown_count=1,
                        click_count=0,
                        last_shown_at=now
                    )
                    db.add(imp)
            
            # タグ別表示カウント更新
            for tag in request.tags:
                tag_lower = tag.lower().strip()
                if not tag_lower:
                    continue
                stmt = select(TagPreference).where(
                    TagPreference.user_id == request.user_id,
                    TagPreference.tag == tag_lower
                )
                result = await db.execute(stmt)
                pref = result.scalars().first()
                
                if pref:
                    pref.shown_count = (pref.shown_count or 0) + 1
                else:
                    pref = TagPreference(
                        user_id=request.user_id,
                        tag=tag_lower,
                        shown_count=1,
                        click_count=0,
                        total_view_time=0
                    )
                    db.add(pref)
            
            await db.commit()
            return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"インプレッションエラー: {e}")

@app.post("/api/logs/click")
async def api_log_click(request: ClickRequest):
    """
    クリック記録。
    クリックされた漫画IDとタグを記録。
    """
    try:
        async with get_tracking_db_session() as db:
            now = _now_iso()
            
            # 漫画のクリックカウント更新
            stmt = select(Impression).where(
                Impression.user_id == request.user_id,
                Impression.manga_id == request.manga_id
            )
            result = await db.execute(stmt)
            imp = result.scalars().first()
            
            if imp:
                imp.click_count = (imp.click_count or 0) + 1
                imp.last_clicked_at = now
            else:
                imp = Impression(
                    user_id=request.user_id,
                    manga_id=request.manga_id,
                    shown_count=1,
                    click_count=1,
                    last_shown_at=now,
                    last_clicked_at=now
                )
                db.add(imp)
            
            # タグ別クリックカウント更新
            for tag in request.tags:
                tag_lower = tag.lower().strip()
                if not tag_lower:
                    continue
                stmt = select(TagPreference).where(
                    TagPreference.user_id == request.user_id,
                    TagPreference.tag == tag_lower
                )
                result = await db.execute(stmt)
                pref = result.scalars().first()
                
                if pref:
                    pref.click_count = (pref.click_count or 0) + 1
                else:
                    pref = TagPreference(
                        user_id=request.user_id,
                        tag=tag_lower,
                        shown_count=1,
                        click_count=1,
                        total_view_time=0
                    )
                    db.add(pref)
            
            await db.commit()
            return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"クリックログエラー: {e}")

@app.post("/api/logs/search")
async def api_log_search(request: SearchLogRequest):
    """
    検索タグ記録。
    検索に使用したタグを記録。
    """
    try:
        async with get_tracking_db_session() as db:
            now = _now_iso()
            
            for tag in request.tags:
                tag_lower = tag.lower().strip()
                if not tag_lower:
                    continue
                
                stmt = select(SearchHistory).where(
                    SearchHistory.user_id == request.user_id,
                    SearchHistory.tag == tag_lower
                )
                result = await db.execute(stmt)
                hist = result.scalars().first()
                
                if hist:
                    hist.search_count = (hist.search_count or 0) + 1
                    hist.last_searched_at = now
                else:
                    hist = SearchHistory(
                        user_id=request.user_id,
                        tag=tag_lower,
                        search_count=1,
                        last_searched_at=now
                    )
                    db.add(hist)
            
            await db.commit()
            return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"検索ログエラー: {e}")

# =========================
# パーソナライズおすすめシステム
# =========================

# 定数
RECOMMENDATION_AVG_TIME_HIGH = 30  # 高満足度の閾値（秒）
RECOMMENDATION_AVG_TIME_MED = 10   # 普通の閾値（秒）
RECOMMENDATION_AVG_TIME_LOW = 5    # 飛ばし読みの閾値（秒）
RECOMMENDATION_LONG_MANGA_PAGES = 50  # 長編の閾値
RECOMMENDATION_CTR_LOW_THRESHOLD = 0.1  # 低CTR閾値
RECOMMENDATION_CTR_SAFE_THRESHOLD = 0.15  # 安全CTR閾値
RECOMMENDATION_SHOWN_DECAY_START = 3  # 繰り返し抑制開始回数
RECOMMENDATION_SHOWN_DECAY_RATE = 0.7  # 繰り返し抑制係数
RECOMMENDATION_EXPLORATION_RATIO = 0.2  # 多様性枠の割合


async def _get_user_tag_scores(db: AsyncSession, user_id: str) -> Dict[str, float]:
    """ユーザーのタグスコアを計算（検索履歴 + タグ別CTR + 閲覧時間）"""
    scores: Dict[str, float] = {}
    
    # 検索履歴からのスコア
    stmt = select(SearchHistory).where(SearchHistory.user_id == user_id)
    result = await db.execute(stmt)
    for hist in result.scalars():
        tag = hist.tag.lower()
        scores[tag] = scores.get(tag, 0) + (hist.search_count or 0) * 10  # 検索は意思が強い
    
    # タグ別CTRと閲覧時間からのスコア
    stmt = select(TagPreference).where(TagPreference.user_id == user_id)
    result = await db.execute(stmt)
    for pref in result.scalars():
        tag = pref.tag.lower()
        shown = pref.shown_count or 1
        clicked = pref.click_count or 0
        view_time = pref.total_view_time or 0
        
        ctr = clicked / max(shown, 1)
        time_score = view_time / max(shown, 1)
        
        scores[tag] = scores.get(tag, 0) + (ctr * 100) + (time_score * 2)
    
    return scores


async def _get_user_favorite_authors(db: AsyncSession, tracking_db: AsyncSession, user_id: str) -> Dict[str, int]:
    """ユーザーのお気に入り作者を抽出（完読 or 再訪問 or 高avg_time）"""
    authors: Dict[str, int] = {}
    
    # ユーザーの閲覧ログを取得
    stmt = select(UserLog).where(UserLog.user_id == user_id)
    result = await tracking_db.execute(stmt)
    user_logs = {log.manga_id: log for log in result.scalars()}
    
    if not user_logs:
        return authors
    
    # 該当する漫画の情報を取得
    manga_ids = list(user_logs.keys())
    stmt = select(Gallery).where(Gallery.gallery_id.in_(manga_ids))
    result = await db.execute(stmt)
    
    for gallery in result.scalars():
        log = user_logs.get(gallery.gallery_id)
        if not log:
            continue
        
        # タグから作者を抽出
        try:
            tags = json.loads(gallery.tags) if gallery.tags else []
        except (TypeError, json.JSONDecodeError):
            tags = []

        author = None
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("artist:"):
                author = tag[7:].lower()
                break
        
        if not author:
            continue
        
        page_count = gallery.page_count or 0
        read_pages = log.read_pages or 0
        visit_count = log.visit_count or 0
        avg_time = log.avg_time or 0
        
        score = 0
        # 完読（read_pages >= page_count * 0.8 とみなす）
        if page_count > 0 and read_pages >= page_count * 0.8:
            score = max(score, 50)
        # 再訪問
        if visit_count >= 2:
            score = max(score, 30)
        # 高avg_time
        if avg_time >= RECOMMENDATION_AVG_TIME_HIGH:
            score = max(score, 20)
        
        if score > 0:
            authors[author] = max(authors.get(author, 0), score)
    
    return authors


async def _get_low_ctr_authors(db: AsyncSession, tracking_db: AsyncSession, user_id: str) -> Set[str]:
    """低CTRの作者を取得"""
    low_ctr_authors: Set[str] = set()
    
    # インプレッションデータを取得
    stmt = select(Impression).where(Impression.user_id == user_id)
    result = await tracking_db.execute(stmt)
    impressions = {imp.manga_id: imp for imp in result.scalars()}
    
    if not impressions:
        return low_ctr_authors
    
    # 該当する漫画の情報を取得
    manga_ids = list(impressions.keys())
    stmt = select(Gallery).where(Gallery.gallery_id.in_(manga_ids))
    result = await db.execute(stmt)
    
    # 作者ごとのCTR集計
    author_stats: Dict[str, Tuple[int, int]] = {}  # author -> (shown, clicked)
    
    for gallery in result.scalars():
        imp = impressions.get(gallery.gallery_id)
        if not imp:
            continue

        try:
            tags = json.loads(gallery.tags) if gallery.tags else []
        except (TypeError, json.JSONDecodeError):
            tags = []

        author = None
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("artist:"):
                author = tag[7:].lower()
                break
        
        if not author:
            continue
        
        shown, clicked = author_stats.get(author, (0, 0))
        author_stats[author] = (shown + (imp.shown_count or 0), clicked + (imp.click_count or 0))
    
    for author, (shown, clicked) in author_stats.items():
        if shown >= 5:  # 十分なデータがある場合のみ
            ctr = clicked / shown
            if ctr < RECOMMENDATION_CTR_LOW_THRESHOLD:
                low_ctr_authors.add(author)
    
    return low_ctr_authors


async def _get_ranking_bonuses(db: AsyncSession, manga_ids: List[int]) -> Dict[int, int]:
    """ランキングボーナスを一括取得（N+1問題回避）"""
    bonuses: Dict[int, int] = {}
    
    if not manga_ids:
        return bonuses
    
    stmt = select(GalleryRanking).where(GalleryRanking.gallery_id.in_(manga_ids))
    result = await db.execute(stmt)
    
    for ranking in result.scalars():
        manga_id = ranking.gallery_id
        rtype = ranking.ranking_type
        score = ranking.score or 0
        
        if manga_id not in bonuses:
            bonuses[manga_id] = 0
        
        # TOP100内かどうかを判定（scoreが高い = 順位が高い）
        if rtype == "weekly" and score > 0:
            bonuses[manga_id] += 30
        elif rtype == "monthly" and score > 0:
            bonuses[manga_id] += 20
        elif rtype == "yearly" and score > 0:
            bonuses[manga_id] += 10
    
    return bonuses


async def _get_global_manga_stats(tracking_db: AsyncSession, manga_ids: List[int]) -> Dict[int, Tuple[float, float]]:
    """グローバル統計（全ユーザーのCTRと平均閲覧時間）を一括取得"""
    stats: Dict[int, Tuple[float, float]] = {}  # manga_id -> (global_ctr, global_avg_time)
    
    if not manga_ids:
        return stats
    
    # インプレッションからグローバルCTRを計算
    # 漫画ごとの集計CTR
    stmt = select(
        Impression.manga_id,
        func.sum(Impression.shown_count).label('total_shown'),
        func.sum(Impression.click_count).label('total_clicked')
    ).where(
        Impression.manga_id.in_(manga_ids)
    ).group_by(Impression.manga_id)
    
    result = await tracking_db.execute(stmt)
    imp_stats = {row[0]: (row[1] or 0, row[2] or 0) for row in result.fetchall()}
    
    # 閲覧ログからグローバル平均閲覧時間を計算
    stmt = select(
        UserLog.manga_id,
        func.avg(UserLog.avg_time).label('avg_time')
    ).where(
        UserLog.manga_id.in_(manga_ids)
    ).group_by(UserLog.manga_id)
    
    result = await tracking_db.execute(stmt)
    time_stats = {row[0]: row[1] or 0 for row in result.fetchall()}
    
    # 統合
    for manga_id in manga_ids:
        shown, clicked = imp_stats.get(manga_id, (0, 0))
        global_ctr = clicked / max(shown, 1) if shown > 0 else 0.15  # デフォルト
        global_avg_time = time_stats.get(manga_id, 15)  # デフォルト15秒
        stats[manga_id] = (global_ctr, global_avg_time)
    
    return stats


async def _get_user_read_manga_ids(tracking_db: AsyncSession, user_id: str) -> Set[int]:
    """ユーザーが閲覧済みの漫画IDセットを取得"""
    stmt = select(UserLog.manga_id).where(UserLog.user_id == user_id)
    result = await tracking_db.execute(stmt)
    return {row[0] for row in result.fetchall()}


async def _get_skipped_manga_ids(tracking_db: AsyncSession, user_id: str) -> Set[int]:
    """スルーされた漫画IDセットを取得（shown > 5 and click = 0）"""
    stmt = select(Impression.manga_id).where(
        Impression.user_id == user_id,
        Impression.shown_count > 5,
        Impression.click_count == 0
    )
    result = await tracking_db.execute(stmt)
    return {row[0] for row in result.fetchall()}


async def _get_shown_counts(tracking_db: AsyncSession, user_id: str) -> Dict[int, int]:
    """漫画ごとの表示回数を取得"""
    stmt = select(Impression).where(Impression.user_id == user_id)
    result = await tracking_db.execute(stmt)
    return {imp.manga_id: imp.shown_count or 0 for imp in result.scalars()}


async def _get_continue_reading_candidates(
    db: AsyncSession, 
    tracking_db: AsyncSession, 
    user_id: str
) -> List[Tuple[int, int]]:
    """「続きを読む」候補を取得（manga_id, bonus）"""
    candidates: List[Tuple[int, int]] = []
    
    # avg_time高 + 未完読のログを取得
    stmt = select(UserLog).where(
        UserLog.user_id == user_id,
        UserLog.avg_time >= 20
    )
    result = await tracking_db.execute(stmt)
    logs = {log.manga_id: log for log in result.scalars()}
    
    if not logs:
        return candidates
    
    # 該当漫画の情報を取得
    manga_ids = list(logs.keys())
    stmt = select(Gallery).where(Gallery.gallery_id.in_(manga_ids))
    result = await db.execute(stmt)
    
    for gallery in result.scalars():
        log = logs.get(gallery.gallery_id)
        if not log:
            continue
        
        page_count = gallery.page_count or 0
        read_pages = log.read_pages or 0
        
        # 長編 + 未完読
        if page_count > RECOMMENDATION_LONG_MANGA_PAGES and read_pages < page_count:
            candidates.append((gallery.gallery_id, 100))
    
    return candidates


async def _get_user_known_tags(tracking_db: AsyncSession, user_id: str) -> Set[str]:
    """ユーザーが見たことあるタグを取得"""
    known_tags: Set[str] = set()
    
    stmt = select(TagPreference.tag).where(TagPreference.user_id == user_id)
    result = await tracking_db.execute(stmt)
    for row in result.fetchall():
        known_tags.add(row[0].lower())
    
    stmt = select(SearchHistory.tag).where(SearchHistory.user_id == user_id)
    result = await tracking_db.execute(stmt)
    for row in result.fetchall():
        known_tags.add(row[0].lower())
    
    return known_tags


async def _get_user_search_counts(tracking_db: AsyncSession, user_id: str) -> Dict[str, int]:
    """ユーザーの検索回数履歴を取得"""
    counts: Dict[str, int] = {}
    stmt = select(SearchHistory).where(SearchHistory.user_id == user_id)
    result = await tracking_db.execute(stmt)
    for row in result.scalars():
        if row.tag:
            counts[row.tag.lower()] = row.search_count or 0
    return counts



def _calculate_quality_factor(avg_time: float, ctr: float) -> float:
    """品質係数を計算（サムネ詐欺 / 隠れた名作判定）"""
    # CTR高 + avg_time低 = サムネ詐欺
    if ctr > 0.3 and avg_time < RECOMMENDATION_AVG_TIME_LOW:
        return 0.5
    # CTR低 + avg_time高 = 隠れた名作
    if ctr < 0.15 and avg_time > RECOMMENDATION_AVG_TIME_HIGH:
        return 2.0
    return 1.0


def _calculate_freshness_factor(created_at_unix: Optional[int]) -> float:
    """新鮮度係数を計算"""
    if not created_at_unix:
        return 1.0
    
    now = int(time.time())
    age_days = (now - created_at_unix) / 86400
    
    if age_days <= 7:
        return 1.5
    elif age_days <= 30:
        return 1.2
    return 1.0


def _calculate_shown_decay(shown_count: int) -> float:
    """繰り返し表示抑制係数を計算"""
    if shown_count <= RECOMMENDATION_SHOWN_DECAY_START:
        return 1.0
    excess = shown_count - RECOMMENDATION_SHOWN_DECAY_START
    return RECOMMENDATION_SHOWN_DECAY_RATE ** excess


@app.get("/api/recommendations/personal")
async def api_recommendations_personal(
    user_id: str,
    limit: int = 20,
    exclude_tag: Optional[str] = None
):
    """
    パーソナライズおすすめAPI
    8つのスコアリング要素を考慮:
    1. タグスコア（検索履歴 + CTR + 閲覧時間）
    2. 作者スコア（お気に入り作者ボーナス / 低CTR作者ペナルティ）
    3. ランキングスコア（週間/月間/年間TOP100）
    4. 品質係数（サムネ詐欺 / 隠れた名作）
    5. 新鮮度係数
    6. 繰り返し表示抑制
    7. 多様性確保（Exploration枠）
    8. 続きを読むボーナス
    """
    
    # パーソナライズ設定チェック
    if not global_state.personalization_enabled:
        # パーソナライズ機能が無効の場合は空のレスポンスを返す
        # フロントエンドはこれを受けておすすめセクションを非表示にする
        return {"count": 0, "results": [], "has_personalization": False, "disabled": True}

    try:
        async with get_db_session() as db, get_tracking_db_session() as tracking_db:
            # ユーザーデータ取得（順次実行 - SQLAlchemy非同期セッションは並列不可）
            tag_scores = await _get_user_tag_scores(tracking_db, user_id)
            read_manga_ids = await _get_user_read_manga_ids(tracking_db, user_id)
            skipped_manga_ids = await _get_skipped_manga_ids(tracking_db, user_id)
            shown_counts = await _get_shown_counts(tracking_db, user_id)
            known_tags = await _get_user_known_tags(tracking_db, user_id)
            favorite_authors = await _get_user_favorite_authors(db, tracking_db, user_id)
            low_ctr_authors = await _get_low_ctr_authors(db, tracking_db, user_id)
            continue_reading = await _get_continue_reading_candidates(db, tracking_db, user_id)
            
            continue_reading_map = {m[0]: m[1] for m in continue_reading}
            
            # 除外タグ処理
            exclude_tag_terms = set()
            if exclude_tag:
                for t in exclude_tag.replace(",", " ").split():
                    t = t.strip().lower()
                    if t:
                        exclude_tag_terms.add(t)
            
            # 候補漫画を取得（最近の作品 + ランキング上位から）
            candidate_limit = limit * 10
            stmt = select(Gallery).where(
                Gallery.manga_type.in_(["doujinshi", "manga"])
            ).order_by(Gallery.created_at_unix.desc()).limit(candidate_limit)
            result = await db.execute(stmt)
            candidates = list(result.scalars())
            
            # バッチ取得: N+1問題回避
            candidate_ids = [g.gallery_id for g in candidates]
            ranking_bonuses = await _get_ranking_bonuses(db, candidate_ids)
            global_stats = await _get_global_manga_stats(tracking_db, candidate_ids)
            
            # スコア計算
            scored_results: List[Tuple[float, Dict[str, Any]]] = []
            exploration_pool: List[Dict[str, Any]] = []
            
            for gallery in candidates:
                manga_id = gallery.gallery_id
                
                # 除外チェック
                if manga_id in skipped_manga_ids:
                    continue
                if manga_id in read_manga_ids and manga_id not in continue_reading_map:
                    continue
                
                # タグ解析
                try:
                    tags = json.loads(gallery.tags) if gallery.tags else []
                except (TypeError, json.JSONDecodeError):
                    tags = []
                
                tag_list = [t.lower() for t in tags if isinstance(t, str)]
                
                # 除外タグチェック
                if exclude_tag_terms and any(t in exclude_tag_terms for t in tag_list):
                    continue
                
                # 作者抽出
                author = None
                for t in tag_list:
                    if t.startswith("artist:"):
                        author = t[7:]
                        break
                
                # スコア計算
                # 1. タグスコア
                tag_score = sum(tag_scores.get(t, 0) for t in tag_list)
                
                # 2. 作者スコア
                author_score = 0
                if author:
                    author_score = favorite_authors.get(author, 0)
                    if author in low_ctr_authors:
                        author_score -= 30
                
                # 3. ランキングボーナス（バッチ取得済み）
                ranking_score = ranking_bonuses.get(manga_id, 0)
                
                # 4. 続きを読むボーナス
                continue_bonus = continue_reading_map.get(manga_id, 0)
                
                # 5. 品質係数（グローバル統計から計算）
                global_ctr, global_avg_time = global_stats.get(manga_id, (0.15, 15))
                quality = _calculate_quality_factor(global_avg_time, global_ctr)
                
                # 6. 新鮮度
                freshness = _calculate_freshness_factor(gallery.created_at_unix)
                
                # 7. 繰り返し抑制
                shown = shown_counts.get(manga_id, 0)
                decay = _calculate_shown_decay(shown)
                
                # 最終スコア
                base_score = tag_score + author_score + ranking_score + continue_bonus
                final_score = base_score * quality * freshness * decay
                
                # 結果オブジェクト作成
                result_obj = {
                    "gallery_id": gallery.gallery_id,
                    "japanese_title": gallery.japanese_title,
                    "tags": tags,
                    "page_count": gallery.page_count,
                    "created_at": gallery.created_at,
                    "manga_type": gallery.manga_type,
                    "score": final_score,
                    "files": gallery.files
                }
                
                # 多様性枠候補チェック（未知のタグを含む）
                has_new_tag = any(t not in known_tags for t in tag_list if not t.startswith("artist:"))
                
                if has_new_tag and ranking_score > 0:
                    exploration_pool.append(result_obj)
                
                if final_score > 0:
                    scored_results.append((final_score, result_obj))
            
            # スコア順ソート
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # 多様性枠の計算
            exploration_count = int(limit * RECOMMENDATION_EXPLORATION_RATIO)
            main_count = limit - exploration_count
            
            # メイン枠
            main_results = [r[1] for r in scored_results[:main_count]]
            
            # Exploration枠（メインに含まれていないものから）
            main_ids = {r["gallery_id"] for r in main_results}
            exploration_candidates = [r for r in exploration_pool if r["gallery_id"] not in main_ids]
            
            random.shuffle(exploration_candidates)
            exploration_results = exploration_candidates[:exploration_count]
            
            # 結果結合
            final_results = main_results + exploration_results
            
            # image_urls追加
            await _process_results_with_image_urls(final_results)
            
            return {
                "count": len(final_results),
                "results": final_results,
                "has_personalization": len(tag_scores) > 0 or len(favorite_authors) > 0
            }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"おすすめ取得エラー: {e}")

# =========================
# ランキング系API
# =========================
@app.get("/api/rankings")
async def get_rankings(
    ranking_type: str = "daily",
    limit: int = 50,
    offset: int = 0,
    tag: Optional[str] = None,
):
    """
    ランキングデータを取得するAPI - searchエンドポイントにリダイレクト
    ranking_type: 'daily', 'weekly', 'monthly', 'yearly', 'all_time'
    tag: タグでフィルタリング（オプション）
    """
    if limit > 200:
        limit = 200
        
    try:
        async with get_db_session() as db:
            # ランキングデータを取得
            ranking_ids = await _load_ranking_ids(ranking_type)
                
            # タグフィルタリング
            if tag:
                known_tags = await _get_known_tag_set(db)
                tag_terms = _parse_tag_terms(tag, known_tags)
                if tag_terms:
                    tag_exists_clauses, tag_params = _build_tag_exists_clause("g", tag_terms)
                    tag_query = f"""
                        SELECT DISTINCT g.gallery_id
                        FROM galleries AS g
                        WHERE {' AND '.join(tag_exists_clauses)}
                    """
                    tag_result = await db.execute(text(tag_query), tag_params)
                    tag_gallery_ids = {row.gallery_id for row in tag_result.fetchall()}
                    filtered_ranking_ids = [gid for gid in ranking_ids if gid in tag_gallery_ids]
                else:
                    filtered_ranking_ids = []
            else:
                filtered_ranking_ids = ranking_ids
            
            # ページネーションを適用
            start_idx = offset
            end_idx = min(start_idx + limit, len(filtered_ranking_ids))
            paginated_ids = filtered_ranking_ids[start_idx:end_idx]
            
            # ギャラリー情報を取得
            if paginated_ids:
                placeholders = ', '.join([f':id_{i}' for i in range(len(paginated_ids))])
                params_db = {f'id_{i}': gallery_id for i, gallery_id in enumerate(paginated_ids)}
                order_case_parts = [f"WHEN :id_{i} THEN {i}" for i in range(len(paginated_ids))]
                order_case = "CASE g.gallery_id " + " ".join(order_case_parts) + f" ELSE {len(paginated_ids)} END"
                
                query = f"""
                    SELECT
                        g.gallery_id,
                        g.japanese_title,
                        g.tags,
                        g.characters,
                        g.files,
                        g.page_count,
                        g.created_at,
                        g.created_at_unix
                    FROM galleries AS g
                    WHERE g.gallery_id IN ({placeholders})
                    ORDER BY {order_case}
                """
                
                result = await db.execute(text(query), params_db)
                rows = result.fetchall()
                
                # 画像URLを並列で取得（高速化）
                rankings = []
                geturl_tasks = []
                
                for row in rows:
                    try:
                        files_data = json.loads(row.files) if hasattr(row, 'files') and row.files else []
                        files_list = files_data if isinstance(files_data, list) else []
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        files_list = []
                    gallery_info = {"gallery_id": row.gallery_id, "files": files_list}
                    geturl_tasks.append(geturl(gallery_info))
                
                # asyncio.gatherで全てのgeturlコルーチンを並列実行
                image_urls_results = await asyncio.gather(*geturl_tasks, return_exceptions=True)
                
                # 結果をマージしてレスポンスを構築
                for rank_position, (row, image_urls) in enumerate(zip(rows, image_urls_results), start=offset + 1):
                    ranking_data = {
                        "gallery_id": row.gallery_id,
                        "ranking_type": ranking_type,
                        "rank": rank_position,
                        "japanese_title": row.japanese_title,
                        "tags": row.tags,
                        "characters": row.characters,
                        "page_count": row.page_count,
                        "created_at": row.created_at,
                        "created_at_unix": row.created_at_unix,
                        "image_urls": image_urls if not isinstance(image_urls, Exception) else []
                    }
                    rankings.append(ranking_data)
                
                return {
                    "rankings": rankings,
                    "count": len(rankings),
                    "total": len(filtered_ranking_ids),
                    "has_more": (offset + limit) < len(filtered_ranking_ids)
                }
            else:
                return {
                    "rankings": [],
                    "count": 0,
                    "total": 0,
                    "has_more": False
                }
    except Exception as e:
        print(f"ランキングデータの取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ランキングデータの取得エラー: {str(e)}")

# =========================
# 定期同期タスク
# =========================
async def hourly_sync_task():
    while True:
        try:
            current_time = datetime.now()
            next_hour = (current_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            wait_seconds = (next_hour - current_time).total_seconds()
            print(f"次回の ImageUriResolver 同期は {next_hour.strftime('%Y-%m-%d %H:%M:%S')} に実行")
            await asyncio.sleep(wait_seconds)
            print("ImageUriResolver.async_synchronize() 実行")
            await ImageUriResolver.async_synchronize()
            print("ImageUriResolver 同期完了")
        except Exception as e:
            print(f"同期中にエラー: {str(e)}")
            await asyncio.sleep(3600)

async def daily_ranking_update_task():
    """日次ランキング更新タスク（*_ids.txtファイルを定期的に更新）"""
    while True:
        try:
            # 毎日午前2時に実行
            now = datetime.now()
            next_update = now.replace(hour=2, minute=0, second=0, microsecond=0)
            if next_update <= now:
                next_update = next_update + timedelta(days=1)
            
            wait_seconds = (next_update - now).total_seconds()
            print(f"次回のランキングファイル更新は {next_update.strftime('%Y-%m-%d %H:%M:%S')} に実行")
            await asyncio.sleep(wait_seconds)
            
            print("ランキングファイル更新開始")
            try:
                # hitomi.pyのdownload_all_popular_files()を実行
                import sys
                
                # スクリプトを非同期で実行
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "scraper/hitomi.py",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=300)  # 5分タイムアウト
                    stdout = stdout_bytes.decode('utf-8', errors='replace') if stdout_bytes else ""
                    stderr = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ""
                    
                    if process.returncode == 0:
                        print("ランキングファイル更新完了")
                        print(f"出力: {stdout}")
                    else:
                        print(f"ランキングファイル更新エラー: {stderr}")
                except asyncio.TimeoutError:
                    process.terminate()
                    await process.wait()
                    print("ランキングファイル更新がタイムアウトしました")
            except Exception as e:
                print(f"ランキングファイル更新中にエラー: {str(e)}")
        except Exception as e:
            print(f"ランキング更新タスク全体でエラー: {str(e)}")
            await asyncio.sleep(3600)  # エラーの場合は1時間待って再試行

async def _precache_search_counts():
    """
    起動時に人気の検索条件に対するCOUNT結果をプリキャッシュする。
    これにより、ユーザーの初回リクエストを高速化。
    """
    try:
        print("[precache] 検索結果COUNTプリキャッシュ開始...")
        start_time = time.time()
        
        # 人気のmin_pages値をプリキャッシュ
        popular_min_pages = [10, 15, 20, 30, 50]
        
        # 1. ページ数範囲のギャラリーIDセットをプリキャッシュ（ランキング+min_pages用）
        for min_pages in popular_min_pages:
            try:
                await _get_page_range_gallery_ids(min_pages, 10_000)
            except Exception as e:
                print(f"[precache] page_range min_pages={min_pages}のプリキャッシュ失敗: {e}")
        
        print(f"[precache] ページ数範囲キャッシュ完了 ({time.time() - start_time:.1f}秒)")
        
        # 2. COUNTクエリのプリキャッシュ（通常検索用）
        async with get_db_session() as db:
            for min_pages in [None] + popular_min_pages:
                try:
                    cache_key = _make_search_count_cache_key(
                        None, None, None, None, None, min_pages, None
                    )
                    
                    # 既にキャッシュされている場合はスキップ
                    if cache_key in _SEARCH_COUNT_CACHE:
                        continue
                    
                    count_query = """
                        SELECT COUNT(*) as total_count
                        FROM galleries AS g
                        WHERE g.manga_type IN ('doujinshi', 'manga')
                    """
                    count_params = {}
                    
                    if min_pages is not None:
                        count_params["min_pages"] = min_pages
                        count_params["max_pages"] = 10_000
                        count_query += " AND g.page_count BETWEEN :min_pages AND :max_pages"
                    
                    count_result = await db.execute(text(count_query), count_params)
                    total_count = count_result.scalar() or 0
                    
                    _SEARCH_COUNT_CACHE[cache_key] = total_count
                except Exception as e:
                    print(f"[precache] min_pages={min_pages}のプリキャッシュ失敗: {e}")
        
        elapsed = time.time() - start_time
        print(f"[precache] 全プリキャッシュ完了 ({elapsed:.1f}秒)")
    except Exception as e:
        print(f"[precache] プリキャッシュエラー: {e}")

# =========================
# ライフサイクル
# =========================
@app.on_event("startup")
async def startup_event():
    global global_session

    # DB 初期化
    await init_database()
    print("メインDB初期化完了")
    # バックフィル処理は同期的に実行（バックグラウンド実行すると他の操作と競合してDB破損の原因になる）
    await _backfill_database_data()

    await init_tracking_database()
    print("トラッキングDB初期化完了")

    # ImageUriResolver 初期化
    try:
        print("ImageUriResolver 初期化")
        await ImageUriResolver.async_synchronize()
        print("ImageUriResolver 初期化完了")
    except Exception as e:
        print(f"ImageUriResolver 初期化エラー: {str(e)}")

    # HTTP セッション（高速化最適化済み）
    connector = aiohttp.TCPConnector(
        limit=200,                    # 全体接続数を増加
        limit_per_host=50,            # 同一ホストへの同時接続数を増加
        ttl_dns_cache=600,            # DNSキャッシュを10分に延長
        use_dns_cache=True,
        keepalive_timeout=120,        # Keep-alive延長（接続再利用を促進）
        force_close=False,
        enable_cleanup_closed=True,   # 閉じた接続のクリーンアップ
    )
    global_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(
            total=60,       # 全体タイムアウト
            connect=10,     # 接続タイムアウト
            sock_read=30,   # 読み取りタイムアウト
        ),
        headers=_build_headers(),
        connector=connector,
    )
    global_state.global_session = global_session

    # 事前ウォームアップ
    try:
        await _warmup_connections(global_session)
    except Exception as e:
        print(f"事前接続エラー: {str(e)}")

    # 同期スケジューラ開始
    global_state.scheduler_task = asyncio.create_task(hourly_sync_task())
    print("同期スケジューラ開始")
    
    # ランキング更新スケジューラ開始
    global_state.ranking_scheduler_task = asyncio.create_task(daily_ranking_update_task())
    print("ランキングファイル更新スケジューラ開始")
    
    # 検索結果COUNTクエリのプリキャッシュ（初回リクエスト高速化）
    asyncio.create_task(_precache_search_counts())

@app.on_event("shutdown")
async def shutdown_event():
    await global_state.cleanup()

# =========================
# エントリポイント
# =========================
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run FastAPI server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--log", type=str, choices=["critical", "error", "warning", "info", "debug", "trace"],
                        default="info", help="Logging level (default: info)")
    parser.add_argument("--disable-personalization", action="store_true", help="Disable personalized recommendations")
    args = parser.parse_args()

    # パーソナライズ設定を反映
    if args.disable_personalization:
        global_state.personalization_enabled = False
        print("Personalized recommendations disabled.")

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log)