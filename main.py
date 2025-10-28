from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import aiohttp
import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse
from types import SimpleNamespace
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, select
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import text
import re
import shlex
import json
import time
from datetime import datetime, timedelta
from lib import ImageUriResolver
import logging

# =========================
# ログ設定
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# グローバルHTTPセッション
# =========================
global_session: Optional[aiohttp.ClientSession] = None

# =========================
# データベース設定
# =========================
DB_FILE = "sa.db"
TRACKING_DB_FILE = "tracking.db"

# メインデータベース（ギャラリー用）
engine = create_engine(
    f"sqlite:///db/{DB_FILE}",
    echo=False,
    connect_args={
        "check_same_thread": False,
        "timeout": 20,
        "isolation_level": None,  # autocommit mode
    },
    pool_pre_ping=True,
    pool_recycle=3600,
)

# トラッキング用データベース
tracking_engine = create_engine(
    f"sqlite:///db/{TRACKING_DB_FILE}",
    echo=False,
    connect_args={
        "check_same_thread": False,
        "timeout": 20,
        "isolation_level": None,  # autocommit mode
    },
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False)
TrackingSessionLocal = sessionmaker(autocommit=False, autoflush=False)

def get_db_session():
    return SessionLocal(bind=engine)

def get_tracking_db_session():
    return TrackingSessionLocal(bind=tracking_engine)

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

    def __repr__(self):
        return f"<Gallery(id={self.gallery_id}, title='{self.japanese_title}')>"

# ---- トラッキング ----
class UserSession(TrackingBase):
    __tablename__ = 'user_sessions'
    session_id = Column(String, primary_key=True)
    fingerprint_hash = Column(String, nullable=False)
    user_agent = Column(Text)
    ip_address = Column(String)
    created_at = Column(String)
    last_activity = Column(String)

class PageView(TrackingBase):
    __tablename__ = 'page_views'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    page_url = Column(String, nullable=False)
    page_title = Column(String)
    referrer = Column(String)
    view_start = Column(String)
    view_end = Column(String)
    time_on_page = Column(Integer)  # 秒
    scroll_depth_max = Column(Integer)

class UserEvent(TrackingBase):
    __tablename__ = 'user_events'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    page_view_id = Column(Integer)
    event_type = Column(String, nullable=False)  # 'click', 'mouse_move', 'scroll'
    element_selector = Column(String)
    element_text = Column(String)
    x_position = Column(Integer)
    y_position = Column(Integer)
    scroll_direction = Column(String)  # 'up' or 'down'
    scroll_speed = Column(Integer)     # ピクセル/秒
    timestamp = Column(String)

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

class MultipartDownloadError(Exception):
    """Raised when multi-part download cannot be completed."""

# =========================
# DB 初期化
# =========================

def init_database():
    """
    - 通常テーブル作成
    - SQLite PRAGMA 最適化
    - FTS5 仮想テーブル + 同期トリガー
    - gallery_tags 正規化テーブル + 強インデックス
    - tag_stats 集約テーブル（インクリメンタル更新）
    - 初回/不一致時の REBUILD と BACKFILL
    """
    import os
    import shutil

    global engine

    # 接続確認
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.warning(f"データベース再作成: {e}")
        engine.dispose()
        # バックアップして削除
        if os.path.exists(f"db/{DB_FILE}"):
            try:
                shutil.copy2(f"db/{DB_FILE}", f"db/{DB_FILE}.corrupt_backup")
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
            except Exception:
                pass
        engine = create_engine(
            f"sqlite:///db/{DB_FILE}",
            echo=False,
            connect_args={
                "check_same_thread": False,
                "timeout": 20,
                "isolation_level": None,
            },
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    Base.metadata.create_all(bind=engine)

    with engine.connect() as conn:
        # PRAGMA
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA synchronous=NORMAL"))
        conn.execute(text("PRAGMA cache_size=10000"))
        conn.execute(text("PRAGMA temp_store=MEMORY"))
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.execute(text("PRAGMA mmap_size=268435456"))  # 256MB

        # 重要インデックス
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_galleries_created ON galleries(created_at DESC, gallery_id DESC)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_galleries_type_created_id ON galleries(manga_type, created_at DESC, gallery_id DESC)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_galleries_characters ON galleries(characters)"))

        # 正規化タグテーブル
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS gallery_tags (
                gallery_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (gallery_id, tag),
                FOREIGN KEY (gallery_id) REFERENCES galleries(gallery_id) ON DELETE CASCADE
            )
            """
        ))
        # 交差（AND）検索向けの強インデックス
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gallery_tags_tag_gallery ON gallery_tags(tag, gallery_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gallery_tags_gallery_tag ON gallery_tags(gallery_id, tag)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gallery_tags_tag ON gallery_tags(tag)"))

        # FTS5
        conn.execute(text(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS galleries_fts USING fts5(
                japanese_title,
                tags,
                characters,
                content='galleries',
                content_rowid='gallery_id',
                tokenize = 'unicode61 remove_diacritics 2 tokenchars ''-_+&/#:."()[]{}'''
            )
            """
        ))

        # FTS 同期トリガ
        conn.execute(text(
            """
            CREATE TRIGGER IF NOT EXISTS galleries_ai AFTER INSERT ON galleries BEGIN
                INSERT INTO galleries_fts(rowid, japanese_title, tags, characters)
                VALUES (new.gallery_id, new.japanese_title, new.tags, new.characters);
            END;
            """
        ))
        conn.execute(text(
            """
            CREATE TRIGGER IF NOT EXISTS galleries_ad AFTER DELETE ON galleries BEGIN
                DELETE FROM galleries_fts WHERE rowid = old.gallery_id;
            END;
            """
        ))
        conn.execute(text(
            """
            CREATE TRIGGER IF NOT EXISTS galleries_au AFTER UPDATE ON galleries BEGIN
                UPDATE galleries_fts
                SET japanese_title = new.japanese_title,
                    tags           = new.tags,
                    characters     = new.characters
                WHERE rowid = old.gallery_id;
            END;
            """
        ))

        # gallery_tags 同期トリガ（galleries.tags JSON -> gallery_tags）
        conn.execute(text(
            """
            CREATE TRIGGER IF NOT EXISTS gallery_tags_ai AFTER INSERT ON galleries BEGIN
                INSERT OR IGNORE INTO gallery_tags(gallery_id, tag)
                SELECT NEW.gallery_id, LOWER(TRIM(value))
                FROM json_each(CASE WHEN json_valid(NEW.tags) THEN NEW.tags ELSE '[]' END)
                WHERE value IS NOT NULL AND TRIM(value) <> '';
            END;
            """
        ))
        conn.execute(text(
            """
            CREATE TRIGGER IF NOT EXISTS gallery_tags_ad AFTER DELETE ON galleries BEGIN
                DELETE FROM gallery_tags WHERE gallery_id = OLD.gallery_id;
            END;
            """
        ))
        conn.execute(text(
            """
            CREATE TRIGGER IF NOT EXISTS gallery_tags_au AFTER UPDATE OF tags ON galleries BEGIN
                DELETE FROM gallery_tags WHERE gallery_id = NEW.gallery_id;
                INSERT OR IGNORE INTO gallery_tags(gallery_id, tag)
                SELECT NEW.gallery_id, LOWER(TRIM(value))
                FROM json_each(CASE WHEN json_valid(NEW.tags) THEN NEW.tags ELSE '[]' END)
                WHERE value IS NOT NULL AND TRIM(value) <> '';
            END;
            """
        ))

        # ---- 集約テーブル: tag_stats（タグ毎の件数を高速に返す）----
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS tag_stats (
                tag TEXT PRIMARY KEY,
                count INTEGER NOT NULL DEFAULT 0
            )
            """
        ))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tag_stats_count_tag ON tag_stats(count DESC, tag ASC)"))

        # gallery_tags 変更に追随するトリガ（増減）
        conn.execute(text(
            """
            CREATE TRIGGER IF NOT EXISTS tag_stats_ins AFTER INSERT ON gallery_tags BEGIN
                INSERT INTO tag_stats(tag, count) VALUES (NEW.tag, 1)
                ON CONFLICT(tag) DO UPDATE SET count = count + 1;
            END;
            """
        ))
        conn.execute(text(
            """
            CREATE TRIGGER IF NOT EXISTS tag_stats_del AFTER DELETE ON gallery_tags BEGIN
                UPDATE tag_stats SET count = MAX(count - 1, 0) WHERE tag = OLD.tag;
            END;
            """
        ))

        # FTS REBUILD 必要時のみ実施
        need_rebuild = conn.execute(text("SELECT COUNT(*) = 0 FROM galleries_fts")).scalar()
        if need_rebuild:
            conn.execute(text("INSERT INTO galleries_fts(galleries_fts) VALUES('rebuild')"))

        # gallery_tags 初期同期
        need_tag_sync = conn.execute(text("SELECT COUNT(*) = 0 FROM gallery_tags")).scalar()
        if need_tag_sync:
            conn.execute(text(
                """
                INSERT OR IGNORE INTO gallery_tags(gallery_id, tag)
                SELECT g.gallery_id, LOWER(TRIM(value))
                FROM galleries AS g,
                     json_each(CASE WHEN json_valid(g.tags) THEN g.tags ELSE '[]' END)
                WHERE value IS NOT NULL AND TRIM(value) <> ''
                """
            ))

        # tag_stats 初期バックフィル（空の時のみ）
        need_tag_stats_backfill = conn.execute(text("SELECT COUNT(*) = 0 FROM tag_stats")).scalar()
        if need_tag_stats_backfill:
            conn.execute(text(
                """
                INSERT INTO tag_stats(tag, count)
                SELECT tag, COUNT(*) FROM gallery_tags GROUP BY tag
                """
            ))

        # 統計最適化
        conn.execute(text("ANALYZE"))
        conn.execute(text("PRAGMA optimize"))

def init_tracking_database():
    import os
    import shutil

    global tracking_engine
    try:
        with tracking_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.warning(f"トラッキングDB再作成: {e}")
        tracking_engine.dispose()
        if os.path.exists(f"db/{TRACKING_DB_FILE}"):
            try:
                shutil.copy2(f"db/{TRACKING_DB_FILE}", f"db/{TRACKING_DB_FILE}.corrupt_backup")
            except Exception:
                pass
            for suffix in ["-wal", "-shm"]:
                p = f"db/{TRACKING_DB_FILE}{suffix}"
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            try:
                os.remove(f"db/{TRACKING_DB_FILE}")
            except Exception:
                pass
        tracking_engine = create_engine(
            f"sqlite:///db/{TRACKING_DB_FILE}",
            echo=False,
            connect_args={
                "check_same_thread": False,
                "timeout": 20,
                "isolation_level": None,
            },
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    TrackingBase.metadata.create_all(bind=tracking_engine)

    with tracking_engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA synchronous=NORMAL"))
        conn.execute(text("PRAGMA cache_size=10000"))
        conn.execute(text("PRAGMA temp_store=MEMORY"))
        conn.execute(text("PRAGMA foreign_keys=ON"))

        # インデックス
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_sessions_fingerprint ON user_sessions(fingerprint_hash)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_page_views_session_id ON page_views(session_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_events_session_id ON user_events(session_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_events_page_view_id ON user_events(page_view_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_events_type ON user_events(event_type)"))

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


def _get_known_tag_set(db_session) -> Set[str]:
    global _KNOWN_TAGS_CACHE, _KNOWN_TAGS_FETCHED_AT
    now = time.time()
    if _KNOWN_TAGS_CACHE and now - _KNOWN_TAGS_FETCHED_AT < 300:
        return _KNOWN_TAGS_CACHE

    try:
        rows = db_session.execute(text("SELECT tag FROM tag_stats")).fetchall()
    except Exception as exc:
        logger.warning("タグ一覧の取得に失敗しました: %s", exc)
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

    if known_tags and not separators_present and len(lowered) > 1:
        resolved: List[str] = []
        idx = 0
        total = len(lowered)
        while idx < total:
            match = None
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
)

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
            f"EXISTS (SELECT 1 FROM gallery_tags gt WHERE gt.gallery_id = {alias}.gallery_id AND gt.tag = :{key})"
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
            f"NOT EXISTS (SELECT 1 FROM gallery_tags gt WHERE gt.gallery_id = {alias}.gallery_id AND gt.tag = :{key})"
        )
        params[key] = term
    return where_list, params

def search_galleries_fast(
    db_session,
    title: str = None,
    tag: str = None,
    character: str = None,
    limit: int = 50,
    after_id: int | None = None,
    exclude_tag: str | None = None,
    min_pages: int | None = None,
    max_pages: int | None = None,
    exclude_gallery_ids: Optional[Tuple[int, ...]] = None,
) -> List[Dict[str, Any]]:
    known_tags = _get_known_tag_set(db_session)
    tag_terms = _parse_tag_terms(tag, known_tags)
    exclude_tag_terms = _parse_tag_terms(exclude_tag, known_tags)
    fts = _build_fts_query(title, character)

    def run_query(use_fts: bool) -> List[Dict[str, Any]]:
        params: Dict[str, object] = {"limit": limit}
        joins: List[str] = []
        where_clauses: List[str] = []

        # doujinshi のみ
        where_clauses.append("g.manga_type = 'doujinshi'")

        if after_id is not None:
            where_clauses.append("g.gallery_id > :after_id")
            params["after_id"] = after_id

        if use_fts and fts:
            joins.append("JOIN galleries_fts AS f ON f.rowid = g.gallery_id")
            where_clauses.append("f MATCH :fts")
            params["fts"] = fts
        else:
            # FTS が無い場合は LIKE フォールバック（簡易）
            if title:
                where_clauses.append("g.japanese_title LIKE :title_like")
                params["title_like"] = f"%{title}%"
            if character:
                where_clauses.append("g.characters LIKE :character_like")
                params["character_like"] = f"%{character}%"

        # タグ: EXISTS 連鎖で AND 条件を満たす ID のみに絞り込む（GROUP BY/HAVING 回避）
        if tag_terms:
            exists_clauses, exists_params = _build_tag_exists_clause("g", tag_terms)
            where_clauses.extend(exists_clauses)
            params.update(exists_params)

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
            where_clauses.append(
                "json_array_length(CASE WHEN json_valid(g.files) THEN g.files ELSE '[]' END) BETWEEN :min_pages AND :max_pages"
            )

        # SQL 組み立て
        sql_segments = ["SELECT g.*", "FROM galleries AS g"]
        if joins:
            sql_segments.extend(joins)
        if where_clauses:
            sql_segments.append("WHERE " + " AND ".join(where_clauses))
        sql_segments.append("ORDER BY g.created_at DESC, g.gallery_id DESC")
        sql_segments.append("LIMIT :limit")
        sql = "\n".join(sql_segments)

        result = db_session.execute(text(sql), params)
        return [_serialize_gallery(row) for row in result.mappings()]

    # FTS 優先、失敗時は LIKE へ
    if fts:
        try:
            results = run_query(True)
            if results:
                return results
        except Exception:
            pass
    try:
        return run_query(False)
    except Exception:
        return search_galleries(
            db_session,
            title=title,
            tag=tag,
            character=character,
            limit=limit,
            offset=0,
            exclude_tag=exclude_tag,
        )


# 低速フォールバック（互換維持）
def search_galleries(
    db_session,
    title: str = None,
    tag: str = None,
    character: str = None,
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
    ).where(Gallery.manga_type == 'doujinshi')

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

    result = db_session.execute(stmt)
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


def build_session_tag_profile(
    db_session,
    session_id: str,
    lookback_days: int = SESSION_TAG_LOOKBACK_DAYS,
    max_page_views: int = SESSION_TAG_MAX_PAGE_VIEWS,
) -> Dict[str, float]:
    if not session_id:
        return {}

    try:
        with get_tracking_db_session() as tracking_db:
            query = (
                tracking_db.query(PageView)
                .filter(PageView.session_id == session_id)
                .order_by(PageView.id.desc())
                .limit(max_page_views)
            )
            page_views = list(query)
    except Exception as exc:
        logger.error("セッションプロファイル取得エラー: %s", exc)
        return {}

    if not page_views:
        return {}

    now = datetime.now()
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
        gallery_rows = (
            db_session.query(Gallery.gallery_id, Gallery.tags)
            .filter(Gallery.gallery_id.in_(gallery_ids))
            .all()
        )
    except Exception as exc:
        logger.error("ギャラリータグ取得エラー: %s", exc)
        return {}

    tag_weights: Dict[str, float] = {}
    for row in gallery_rows:
        raw_tags = row.tags
        try:
            tags_data = json.loads(raw_tags) if isinstance(raw_tags, str) else raw_tags
        except (TypeError, json.JSONDecodeError):
            tags_data = []

        if not isinstance(tags_data, list):
            continue

        gallery_weight = gallery_scores.get(row.gallery_id, 0.0)
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


def get_recommended_galleries(
    db_session,
    gallery_id: Optional[int] = None,
    limit: int = 8,
    exclude_tag: Optional[str] = None,
    session_tag_weights: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    target_tags: Tuple[str, ...] = tuple()
    exclude_ids: List[int] = []

    if gallery_id is not None:
        gallery = (
            db_session.query(Gallery)
            .filter(Gallery.gallery_id == gallery_id)
            .first()
        )
        if gallery:
            exclude_ids.append(gallery.gallery_id)
            try:
                raw_tags = json.loads(gallery.tags) if gallery.tags else []
            except (TypeError, json.JSONDecodeError):
                raw_tags = []
            normalized = []
            for tag in raw_tags or []:
                if isinstance(tag, str):
                    normalized.append(tag.strip().lower())
            target_tags = tuple(normalized[:10])  # 上位10件まで

    known_tags = _get_known_tag_set(db_session)
    exclude_terms = _parse_tag_terms(exclude_tag, known_tags)

    params: Dict[str, Any] = {"limit": limit}
    where_clauses = ["g.manga_type = 'doujinshi'"]
    joins = []
    order_clause = "ORDER BY g.created_at DESC, g.gallery_id DESC"

    if target_tags:
        joins.append("JOIN gallery_tags gt ON gt.gallery_id = g.gallery_id")
        tag_placeholders = []
        for idx, tag_value in enumerate(target_tags):
            key = f"rec_tag_{idx}"
            params[key] = tag_value
            tag_placeholders.append(f":{key}")
        where_clauses.append(f"gt.tag IN ({', '.join(tag_placeholders)})")
        order_clause = (
            "ORDER BY COUNT(DISTINCT gt.tag) DESC, g.created_at DESC, g.gallery_id DESC"
        )

    if exclude_terms:
        not_exists, not_params = _build_tag_not_exists_clause("g", exclude_terms)
        where_clauses.extend(not_exists)
        params.update(not_params)

    if exclude_ids:
        placeholders = []
        for idx, gid in enumerate(exclude_ids):
            key = f"rec_exclude_{idx}"
            params[key] = gid
            placeholders.append(f":{key}")
        where_clauses.append(f"g.gallery_id NOT IN ({', '.join(placeholders)})")

    select_clause = "SELECT g.*"
    from_clause = "FROM galleries AS g"
    if joins:
        from_clause += " " + " ".join(joins)

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    group_sql = ""
    if target_tags:
        group_sql = "GROUP BY g.gallery_id"

    sql = "\n".join([select_clause, from_clause, where_sql, group_sql, order_clause, "LIMIT :limit"])

    result = db_session.execute(text(sql), params)
    galleries = [_serialize_gallery(row) for row in result.mappings()]

    if len(galleries) < limit:
        # Fallback: 最近の作品を追加で補完
        remaining = limit - len(galleries)
        fallback_params: Dict[str, Any] = {"limit": remaining}
        fallback_clauses = ["g.manga_type = 'doujinshi'"]
        if exclude_terms:
            not_exists, not_params = _build_tag_not_exists_clause("g", exclude_terms)
            fallback_clauses.extend(not_exists)
            fallback_params.update(not_params)
        if exclude_ids:
            placeholders = []
            for idx, gid in enumerate(exclude_ids):
                key = f"fallback_exclude_{idx}"
                fallback_params[key] = gid
                placeholders.append(f":{key}")
            fallback_clauses.append(f"g.gallery_id NOT IN ({', '.join(placeholders)})")

        fallback_where = "WHERE " + " AND ".join(fallback_clauses)
        fallback_sql = "\n".join(
            [
                "SELECT g.*",
                "FROM galleries AS g",
                fallback_where,
                "ORDER BY g.created_at DESC, g.gallery_id DESC",
                "LIMIT :limit",
            ]
        )
        fallback_rows = db_session.execute(text(fallback_sql), fallback_params)
        fallback_list = [_serialize_gallery(row) for row in fallback_rows.mappings()]
        existing_ids = {g["gallery_id"] for g in galleries}
        for item in fallback_list:
            if item["gallery_id"] not in existing_ids:
                galleries.append(item)
                existing_ids.add(item["gallery_id"])
            if len(galleries) >= limit:
                break

    if session_tag_weights:
        normalized_weights = {
            key.strip().lower(): value
            for key, value in session_tag_weights.items()
            if isinstance(key, str) and key.strip()
        }
        if normalized_weights:
            for idx, item in enumerate(galleries):
                item["_base_order"] = idx
                score = 0.0
                raw_tags = item.get("tags")
                try:
                    tags_list = json.loads(raw_tags) if isinstance(raw_tags, str) else raw_tags
                except (TypeError, json.JSONDecodeError):
                    tags_list = []
                if isinstance(tags_list, list):
                    for tag_value in tags_list:
                        normalized = _normalize_tag_value(tag_value)
                        if normalized:
                            score += normalized_weights.get(normalized, 0.0)
                item["_personal_score"] = score

            galleries.sort(
                key=lambda item: (
                    -item.get("_personal_score", 0.0),
                    item.get("_base_order", 0),
                )
            )

            for item in galleries:
                score = item.pop("_personal_score", None)
                item.pop("_base_order", None)
                if score:
                    item["personal_score"] = round(float(score), 4)

    return galleries

# =========================
# 画像 URL 生成
# =========================

def _derive_filename(url: str) -> str:
    trimmed = url.split("?", 1)[0].rstrip("/")
    candidate = trimmed.split("/")[-1] if trimmed else ""
    return candidate or "download.bin"


def geturl(gi: Dict[str, Any]):
    urls: List[str] = []
    files = gi.get("files", []) or []
    try:
        ImageUriResolver.synchronize()
    except Exception as e:
        logger.error(f"ImageUriResolver 初期化エラー: {e}")
        return []

    for idx, f in enumerate(files):
        image = SimpleNamespace(
            index=idx,
            hash=(f.get("hash") or "").lower(),
            name=f.get("name"),
            width=f.get("width"),
            height=f.get("height"),
            has_avif=bool(f.get("hasavif")),
            has_webp=True,
            has_jxl=False,
        )
        ext = "avif" if image.has_avif else "webp"
        try:
            url_img = ImageUriResolver.get_image_uri(image, ext)  # type: ignore
            urls.append(url_img)
        except Exception as e:
            logger.error(f"画像URL生成エラー: {e}, hash: {image.hash}")
            continue
    return urls

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

class SearchRequest(BaseModel):
    title: Optional[str] = None
    tag: Optional[str] = None
    exclude_tag: Optional[str] = None
    character: Optional[str] = None
    limit: int = 50
    after_id: Optional[int] = None
    min_pages: Optional[int] = None
    max_pages: Optional[int] = None

class SessionRequest(BaseModel):
    session_id: str
    fingerprint_hash: str
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None

class PageViewRequest(BaseModel):
    session_id: str
    page_url: str
    page_title: Optional[str] = None
    referrer: Optional[str] = None
    view_start: Optional[str] = None
    view_end: Optional[str] = None
    time_on_page: Optional[int] = None
    scroll_depth_max: Optional[int] = None

class EventRequest(BaseModel):
    session_id: str
    page_view_id: Optional[int] = None
    event_type: str
    element_selector: Optional[str] = None
    element_text: Optional[str] = None
    x_position: Optional[int] = None
    y_position: Optional[int] = None
    scroll_direction: Optional[str] = None
    scroll_speed: Optional[int] = None
    timestamp: Optional[str] = None

class BatchEventsRequest(BaseModel):
    events: List[EventRequest]

# =========================
# ルータ
# =========================
@app.get("/", response_class=HTMLResponse)
async def read_index():
    return _serve_cached_html("template/index.html")

@app.get("/viewer", response_class=HTMLResponse)
async def read_viewer():
    return _serve_cached_html("template/viewer.html")

@app.get("/recommendations", response_class=HTMLResponse)
async def read_recommendations():
    return _serve_cached_html("template/recommendations.html")

@app.post("/search")
async def search_galleries_endpoint(request: SearchRequest):
    try:
        with get_db_session() as db:
            results = search_galleries_fast(
                db,
                title=request.title,
                tag=request.tag,
                exclude_tag=request.exclude_tag,
                character=request.character,
                limit=request.limit,
                after_id=request.after_id,
                min_pages=request.min_pages,
                max_pages=request.max_pages,
            )

            for result in results:
                try:
                    files_data = json.loads(result["files"]) if isinstance(result.get("files"), str) else result.get("files")
                    files_list = files_data if isinstance(files_data, list) else []
                    gallery_info = {"gallery_id": result["gallery_id"], "files": files_data}
                    result["image_urls"] = geturl(gallery_info)
                    result["page_count"] = len(files_list)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"files 解析エラー: {e}, gallery_id: {result['gallery_id']}")
                    result["image_urls"] = []
                    result["page_count"] = 0
                if "files" in result:
                    del result["files"]

            return {"results": results, "count": len(results), "has_more": len(results) == request.limit}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"検索エラー: {str(e)}")

@app.get("/api/recommendations")
async def api_recommendations(
    gallery_id: Optional[int] = None,
    limit: int = 8,
    exclude_tag: Optional[str] = None,
    session_id: Optional[str] = None,
):
    try:
        with get_db_session() as db:
            session_tag_weights: Optional[Dict[str, float]] = None
            if session_id:
                try:
                    session_tag_weights = build_session_tag_profile(db, session_id)
                except Exception as exc:
                    logger.error("おすすめ個人化プロファイル作成エラー: %s", exc)
                    session_tag_weights = None
            results = get_recommended_galleries(
                db,
                gallery_id=gallery_id,
                limit=limit,
                exclude_tag=exclude_tag,
                session_tag_weights=session_tag_weights,
            )
            payload: List[Dict[str, Any]] = []
            for result in results:
                try:
                    files_data = json.loads(result.get("files")) if isinstance(result.get("files"), str) else result.get("files")
                except (json.JSONDecodeError, TypeError):
                    files_data = []
                files_list = files_data if isinstance(files_data, list) else []
                gallery_info = {"gallery_id": result["gallery_id"], "files": files_list}
                image_urls = geturl(gallery_info)
                payload.append(
                    {
                        **{k: v for k, v in result.items() if k != "files"},
                        "image_urls": image_urls,
                        "page_count": len(files_list),
                    }
                )
            return {"results": payload, "count": len(payload)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"おすすめ取得エラー: {str(e)}")

@app.get("/search")
async def search_galleries_get(
    title: Optional[str] = None,
    tag: Optional[str] = None,
    exclude_tag: Optional[str] = None,
    character: Optional[str] = None,
    limit: int = 50,
    after_id: Optional[int] = None,
    min_pages: Optional[int] = None,
    max_pages: Optional[int] = None,
):
    try:
        with get_db_session() as db:
            results = search_galleries_fast(
                db,
                title=title,
                tag=tag,
                exclude_tag=exclude_tag,
                character=character,
                limit=limit,
                after_id=after_id,
                min_pages=min_pages,
                max_pages=max_pages,
            )

            for result in results:
                try:
                    files_data = json.loads(result["files"]) if isinstance(result.get("files"), str) else result.get("files")
                    files_list = files_data if isinstance(files_data, list) else []
                    gallery_info = {"gallery_id": result["gallery_id"], "files": files_data}
                    result["image_urls"] = geturl(gallery_info)
                    result["page_count"] = len(files_list)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"files 解析エラー: {e}, gallery_id: {result['gallery_id']}")
                    result["image_urls"] = []
                    result["page_count"] = 0
                if "files" in result:
                    del result["files"]

            return {"results": results, "count": len(results), "has_more": len(results) == limit}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"検索エラー: {str(e)}")

@app.get("/proxy/{path:path}")
async def proxy_request(path: str):
    # パスにプロトコルがない場合は https:// を付与
    url = path if path.startswith(("http://", "https://")) else f"https://{path}"
    logger.info(f"プロキシリクエスト: {url}")
    headers = _build_headers()

    # セッションが未初期化でも動くようフォールバック
    session = global_session or aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), headers=headers)
    created_local = session is not global_session

    max_retries = 3
    retry_delay = 1
    try:
        for attempt in range(max_retries + 1):
            try:
                async with session.get(url, headers=headers) as resp:
                    logger.info(f"レスポンス: Status={resp.status}, Content-Type={resp.content_type}")
                    if resp.status >= 500:
                        if attempt < max_retries:
                            await asyncio.sleep(retry_delay * (2 ** attempt))
                            continue
                        raise HTTPException(status_code=resp.status, detail=f"Upstream 5xx: {resp.status}")
                    if resp.status >= 400:
                        raise HTTPException(status_code=resp.status, detail=f"Upstream 4xx: {resp.status}")

                    if resp.content_type and resp.content_type.startswith("image/"):
                        data = await resp.read()
                        return Response(
                            content=data,
                            media_type=resp.content_type,
                            headers={'Cache-Control': 'public, max-age=3600', 'Access-Control-Allow-Origin': '*'},
                        )
                    else:
                        content = await resp.text()
                        return HTMLResponse(content=content)
            except aiohttp.ClientError as e:
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                raise HTTPException(status_code=500, detail=f"Proxy client error: {str(e)}")
    finally:
        if created_local:
            await session.close()

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
    domains = ["a1.gold-usergeneratedcontent.net", "a2.gold-usergeneratedcontent.net"]
    for domain in domains:
        url = f"https://{domain}"
        try:
            async with session.head(url) as resp:
                logger.info(f"事前接続成功: {url} (Status: {resp.status})")
        except Exception as e:
            logger.error(f"事前接続失敗: {url} (Error: {str(e)})")

async def _download_single_url(session: aiohttp.ClientSession, url: str, headers: Dict[str, str]) -> Dict[str, str]:
    try:
        async with session.get(url, headers=headers) as resp:
            content = await resp.text()
            return {"url": url, "status": "success", "content": content, "status_code": str(resp.status)}
    except Exception as exc:
        return {"url": url, "status": "error", "content": str(exc), "status_code": "500"}

@app.post("/download-multiple", response_model=DownloadResponse)
async def download_multiple(request: DownloadRequest):
    headers = _build_headers()
    tasks = [_download_single_url(global_session, url, headers) for url in request.urls]
    results = await asyncio.gather(*tasks)
    return DownloadResponse(results=results)

@app.post("/download-multipart")
async def download_multipart(request: MultipartDownloadRequest):
    if request.chunk_size <= 0:
        raise HTTPException(status_code=400, detail="chunk_size must be > 0")
    if request.max_connections <= 0:
        raise HTTPException(status_code=400, detail="max_connections must be > 0")

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
        with get_db_session() as db:
            gallery = db.query(Gallery).filter(Gallery.gallery_id == gallery_id).first()
            if not gallery:
                raise HTTPException(status_code=404, detail="ギャラリーが見つかりません")
            try:
                files_data = json.loads(gallery.files) if isinstance(gallery.files, str) else gallery.files
                files_list = files_data if isinstance(files_data, list) else []
                gallery_info = {"gallery_id": gallery.gallery_id, "files": files_list}
                image_urls = geturl(gallery_info)
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"files 解析エラー: {e}, gallery_id: {gallery.gallery_id}")
                image_urls = []
                files_list = []
            return {
                "gallery_id": gallery.gallery_id,
                "japanese_title": gallery.japanese_title,
                "tags": gallery.tags,
                "characters": gallery.characters,
                "image_urls": image_urls,
                "manga_type": gallery.manga_type,
                "created_at": gallery.created_at,
                "page_count": len(files_list),
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ギャラリー情報の取得エラー: {str(e)}")

# -------------------------
# タグ API（tag_stats 使用で高速化）
# -------------------------
@app.get("/api/tags")
async def get_tags(limit: int = 100, offset: int = 0, search: Optional[str] = None):
    try:
        with get_db_session() as db:
            params: Dict[str, Any] = {"limit": limit, "offset": offset}
            if search:
                params["search"] = f"%{search.lower()}%"
                query = (
                    "SELECT tag, count FROM tag_stats WHERE tag LIKE :search "
                    "ORDER BY count DESC, tag ASC LIMIT :limit OFFSET :offset"
                )
                total_sql = "SELECT COUNT(*) FROM tag_stats WHERE tag LIKE :search"
            else:
                query = (
                    "SELECT tag, count FROM tag_stats "
                    "ORDER BY count DESC, tag ASC LIMIT :limit OFFSET :offset"
                )
                total_sql = "SELECT COUNT(*) FROM tag_stats"

            total_count = db.execute(text(total_sql), params).scalar()
            rows = db.execute(text(query), params).fetchall()
            tags = [{"tag": r.tag, "count": r.count} for r in rows]
            return {"tags": tags, "total": total_count, "has_more": (offset + limit) < (total_count or 0)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"タグ情報の取得エラー: {str(e)}")

@app.get("/tags", response_class=HTMLResponse)
async def read_tags():
    return _serve_cached_html("template/tags.html")

# =========================
# トラッキング系
# =========================
@app.post("/api/tracking/session")
async def update_session(request: SessionRequest):
    try:
        with get_tracking_db_session() as db:
            existing_session = db.query(UserSession).filter(UserSession.session_id == request.session_id).first()
            now_iso = datetime.now().isoformat()
            if existing_session:
                existing_session.last_activity = now_iso
                if request.user_agent:
                    existing_session.user_agent = request.user_agent
                if request.ip_address:
                    existing_session.ip_address = request.ip_address
                db.commit()
                return {"status": "updated", "session_id": existing_session.session_id}
            else:
                new_session = UserSession(
                    session_id=request.session_id,
                    fingerprint_hash=request.fingerprint_hash,
                    user_agent=request.user_agent,
                    ip_address=request.ip_address,
                    created_at=now_iso,
                    last_activity=now_iso,
                )
                db.add(new_session)
                db.commit()
                return {"status": "created", "session_id": new_session.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"セッション更新エラー: {str(e)}")

@app.post("/api/tracking/page-view")
async def record_page_view(request: PageViewRequest):
    try:
        with get_tracking_db_session() as db:
            session = db.query(UserSession).filter(UserSession.session_id == request.session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="セッションが見つかりません")
            new_page_view = PageView(
                session_id=request.session_id,
                page_url=request.page_url,
                page_title=request.page_title,
                referrer=request.referrer,
                view_start=request.view_start or datetime.now().isoformat(),
                view_end=request.view_end,
                time_on_page=request.time_on_page,
                scroll_depth_max=request.scroll_depth_max,
            )
            db.add(new_page_view)
            db.commit()
            return {"status": "created", "page_view_id": new_page_view.id, "session_id": request.session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ページビュー記録エラー: {str(e)}")

@app.post("/api/tracking/events")
async def record_events(request: BatchEventsRequest):
    try:
        with get_tracking_db_session() as db:
            session_ids = set(event.session_id for event in request.events)
            sessions = db.query(UserSession).filter(UserSession.session_id.in_(session_ids)).all()
            session_dict = {s.session_id: s for s in sessions}
            valid_events = [event for event in request.events if event.session_id in session_dict]
            if not valid_events:
                return {"status": "no_valid_events", "count": 0}

            new_events = []
            for event in valid_events:
                new_event = UserEvent(
                    session_id=event.session_id,
                    page_view_id=event.page_view_id,
                    event_type=event.event_type,
                    element_selector=event.element_selector,
                    element_text=event.element_text,
                    x_position=event.x_position,
                    y_position=event.y_position,
                    scroll_direction=event.scroll_direction,
                    scroll_speed=event.scroll_speed,
                    timestamp=event.timestamp or datetime.now().isoformat(),
                )
                new_events.append(new_event)
            db.add_all(new_events)
            db.commit()
            return {"status": "created", "count": len(new_events), "session_id": valid_events[0].session_id if valid_events else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"イベント記録エラー: {str(e)}")

@app.post("/api/tracking/single-event")
async def record_single_event(request: EventRequest):
    try:
        with get_tracking_db_session() as db:
            session = db.query(UserSession).filter(UserSession.session_id == request.session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="セッションが見つかりません")
            new_event = UserEvent(
                session_id=request.session_id,
                page_view_id=request.page_view_id,
                event_type=request.event_type,
                element_selector=request.element_selector,
                element_text=request.element_text,
                x_position=request.x_position,
                y_position=request.y_position,
                scroll_direction=request.scroll_direction,
                scroll_speed=request.scroll_speed,
                timestamp=request.timestamp or datetime.now().isoformat(),
            )
            db.add(new_event)
            db.commit()
            return {"status": "created", "event_id": new_event.id, "session_id": request.session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"イベント記録エラー: {str(e)}")

# =========================
# 定期同期タスク
# =========================
async def hourly_sync_task():
    while True:
        try:
            current_time = datetime.now()
            next_hour = (current_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            wait_seconds = (next_hour - current_time).total_seconds()
            logger.info(f"次回の ImageUriResolver 同期は {next_hour.strftime('%Y-%m-%d %H:%M:%S')} に実行")
            await asyncio.sleep(wait_seconds)
            logger.info("ImageUriResolver.async_synchronize() 実行")
            await ImageUriResolver.async_synchronize()
            logger.info("ImageUriResolver 同期完了")
        except Exception as e:
            logger.error(f"同期中にエラー: {str(e)}")
            await asyncio.sleep(3600)

# =========================
# ライフサイクル
# =========================
@app.on_event("startup")
async def startup_event():
    global global_session, scheduler_task

    # DB 初期化
    init_database()
    logger.info("メインDB初期化完了")
    init_tracking_database()
    logger.info("トラッキングDB初期化完了")

    # ImageUriResolver 初期化
    try:
        logger.info("ImageUriResolver 初期化")
        await ImageUriResolver.async_synchronize()
        logger.info("ImageUriResolver 初期化完了")
    except Exception as e:
        logger.error(f"ImageUriResolver 初期化エラー: {str(e)}")

    # HTTP セッション
    connector = aiohttp.TCPConnector(
        limit=100,
        limit_per_host=30,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=75,
        force_close=False,
    )
    global_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), headers=_build_headers(), connector=connector)

    # 事前ウォームアップ
    try:
        await _warmup_connections(global_session)
    except Exception as e:
        logger.error(f"事前接続エラー: {str(e)}")

    # 同期スケジューラ開始
    scheduler_task = asyncio.create_task(hourly_sync_task())
    logger.info("同期スケジューラ開始")

@app.on_event("shutdown")
async def shutdown_event():
    global global_session, scheduler_task
    if 'scheduler_task' in globals() and scheduler_task:
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            logger.info("同期スケジューラ停止")
    if global_session:
        await global_session.close()
        logger.info("Global session closed")

# =========================
# エントリポイント
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
