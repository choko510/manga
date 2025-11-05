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
from pydantic import BaseModel, Field
from sqlalchemy import Column, Computed, Integer, String, Text, select
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
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
logging.basicConfig(level=logging.WARNING)
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
engine: AsyncEngine = create_async_engine(
    f"sqlite+aiosqlite:///db/{DB_FILE}",
    echo=False,
    connect_args={
        "timeout": 20,
        "isolation_level": None,  # autocommit mode
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
        "isolation_level": None,  # autocommit mode
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
    page_count = Column(
        Integer,
        Computed("json_array_length(CASE WHEN json_valid(files) THEN files ELSE '[]' END)", persisted=True),
    )
    created_at_unix = Column(
        Integer,
        Computed("CAST(strftime('%s', created_at) AS INTEGER)", persisted=True),
    )

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
TAG_CATEGORIES_FILE = Path("static/tag-categories.json")

# ランキングデータファイル
RANKING_FILES = {
    "daily": "day_ids.txt",
    "weekly": "week_ids.txt",
    "monthly": "month_ids.txt",
    "yearly": "year_ids.txt",
    "all_time": "all_ids.txt"
}

async def _load_ranking_ids(ranking_type: str) -> List[int]:
    """
    ランキングデータファイルからギャラリーIDを読み込む
    ranking_type: 'daily', 'weekly', 'monthly', 'yearly', 'all_time'
    """
    if ranking_type not in RANKING_FILES:
        raise ValueError(f"Invalid ranking type: {ranking_type}")
    
    file_path = RANKING_FILES[ranking_type]
    
    def _read_ids():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [int(line.strip()) for line in f if line.strip().isdigit()]
        except FileNotFoundError:
            logger.error(f"ランキングファイルが見つかりません: {file_path}")
            return []
        except Exception as e:
            logger.error(f"ランキングファイル読み込みエラー: {e}")
            return []
    
    return await asyncio.to_thread(_read_ids)


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
# DB 初期化
# =========================

async def init_database():
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

    global engine, SessionLocal

    # 接続確認
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.warning(f"データベース再作成: {e}")
        await engine.dispose()
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
        engine = create_async_engine(
            f"sqlite+aiosqlite:///db/{DB_FILE}",
            echo=False,
            connect_args={
                "timeout": 20,
                "isolation_level": None,
            },
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with engine.begin() as conn:
        # PRAGMA
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.execute(text("PRAGMA synchronous=NORMAL"))
        await conn.execute(text("PRAGMA cache_size=20000"))
        await conn.execute(text("PRAGMA temp_store=MEMORY"))
        await conn.execute(text("PRAGMA foreign_keys=ON"))
        await conn.execute(text("PRAGMA mmap_size=536870912"))  # 512MB

        # 重要インデックス
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_galleries_created ON galleries(created_at DESC, gallery_id DESC)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_galleries_type_created_id ON galleries(manga_type, created_at DESC, gallery_id DESC)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_galleries_characters ON galleries(characters)"))

        # 正規化タグテーブル
        await conn.execute(text(
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
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gallery_tags_tag_gallery ON gallery_tags(tag, gallery_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gallery_tags_gallery_tag ON gallery_tags(gallery_id, tag)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gallery_tags_tag ON gallery_tags(tag)"))

        # FTS5
        await conn.execute(text(
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
        await conn.execute(text(
            """
            CREATE TRIGGER IF NOT EXISTS galleries_ai AFTER INSERT ON galleries BEGIN
                INSERT INTO galleries_fts(rowid, japanese_title, tags, characters)
                VALUES (new.gallery_id, new.japanese_title, new.tags, new.characters);
            END;
            """
        ))
        await conn.execute(text(
            """
            CREATE TRIGGER IF NOT EXISTS galleries_ad AFTER DELETE ON galleries BEGIN
                DELETE FROM galleries_fts WHERE rowid = old.gallery_id;
            END;
            """
        ))
        await conn.execute(text(
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
        await conn.execute(text(
            """
            CREATE TRIGGER IF NOT EXISTS gallery_tags_ai AFTER INSERT ON galleries BEGIN
                INSERT OR IGNORE INTO gallery_tags(gallery_id, tag)
                SELECT NEW.gallery_id, LOWER(TRIM(value))
                FROM json_each(CASE WHEN json_valid(NEW.tags) THEN NEW.tags ELSE '[]' END)
                WHERE value IS NOT NULL AND TRIM(value) <> '';
            END;
            """
        ))
        await conn.execute(text(
            """
            CREATE TRIGGER IF NOT EXISTS gallery_tags_ad AFTER DELETE ON galleries BEGIN
                DELETE FROM gallery_tags WHERE gallery_id = OLD.gallery_id;
            END;
            """
        ))
        await conn.execute(text(
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
        await conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS tag_stats (
                tag TEXT PRIMARY KEY,
                count INTEGER NOT NULL DEFAULT 0
            )
            """
        ))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tag_stats_count_tag ON tag_stats(count DESC, tag ASC)"))

        # gallery_tags 変更に追随するトリガ（増減）
        await conn.execute(text(
            """
            CREATE TRIGGER IF NOT EXISTS tag_stats_ins AFTER INSERT ON gallery_tags BEGIN
                INSERT INTO tag_stats(tag, count) VALUES (NEW.tag, 1)
                ON CONFLICT(tag) DO UPDATE SET count = count + 1;
            END;
            """
        ))
        await conn.execute(text(
            """
            CREATE TRIGGER IF NOT EXISTS tag_stats_del AFTER DELETE ON gallery_tags BEGIN
                UPDATE tag_stats SET count = MAX(count - 1, 0) WHERE tag = OLD.tag;
            END;
            """
        ))

        # FTS REBUILD 必要時のみ実施
        rebuild_result = await conn.execute(text("SELECT COUNT(*) = 0 FROM galleries_fts"))
        need_rebuild = rebuild_result.scalar()
        if need_rebuild:
            await conn.execute(text("INSERT INTO galleries_fts(galleries_fts) VALUES('rebuild')"))

        # gallery_tags 初期同期
        tag_sync_result = await conn.execute(text("SELECT COUNT(*) = 0 FROM gallery_tags"))
        need_tag_sync = tag_sync_result.scalar()
        if need_tag_sync:
            await conn.execute(text(
                """
                INSERT OR IGNORE INTO gallery_tags(gallery_id, tag)
                SELECT g.gallery_id, LOWER(TRIM(value))
                FROM galleries AS g,
                     json_each(CASE WHEN json_valid(g.tags) THEN g.tags ELSE '[]' END)
                WHERE value IS NOT NULL AND TRIM(value) <> ''
                """
            ))

        # tag_stats 初期バックフィル（空の時のみ）
        tag_stats_result = await conn.execute(text("SELECT COUNT(*) = 0 FROM tag_stats"))
        need_tag_stats_backfill = tag_stats_result.scalar()
        if need_tag_stats_backfill:
            await conn.execute(text(
                """
                INSERT INTO tag_stats(tag, count)
                SELECT tag, COUNT(*) FROM gallery_tags GROUP BY tag
                """
            ))

        # ---- ランキングテーブル ----
        await conn.execute(text(
            """
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
            """
        ))
        
        # ランキングテーブルのインデックス
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gallery_rankings_type_score ON gallery_rankings(ranking_type, score DESC)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gallery_rankings_gallery_id ON gallery_rankings(gallery_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gallery_rankings_last_updated ON gallery_rankings(last_updated)"))

        # 統計最適化
        await conn.execute(text("ANALYZE"))
        await conn.execute(text("PRAGMA optimize"))

async def init_tracking_database():
    import os
    import shutil

    global tracking_engine, TrackingSessionLocal
    try:
        async with tracking_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.warning(f"トラッキングDB再作成: {e}")
        await tracking_engine.dispose()
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
        tracking_engine = create_async_engine(
            f"sqlite+aiosqlite:///db/{TRACKING_DB_FILE}",
            echo=False,
            connect_args={
                "timeout": 20,
                "isolation_level": None,
            },
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        TrackingSessionLocal = async_sessionmaker(bind=tracking_engine, expire_on_commit=False)

    async with tracking_engine.begin() as conn:
        await conn.run_sync(TrackingBase.metadata.create_all)

    async with tracking_engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.execute(text("PRAGMA synchronous=NORMAL"))
        await conn.execute(text("PRAGMA cache_size=10000"))
        await conn.execute(text("PRAGMA temp_store=MEMORY"))
        await conn.execute(text("PRAGMA foreign_keys=ON"))

        # インデックス
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_sessions_fingerprint ON user_sessions(fingerprint_hash)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_page_views_session_id ON page_views(session_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_events_session_id ON user_events(session_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_events_page_view_id ON user_events(page_view_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_events_type ON user_events(event_type)"))

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


async def _get_known_tag_set(db_session: AsyncSession) -> Set[str]:
    global _KNOWN_TAGS_CACHE, _KNOWN_TAGS_FETCHED_AT
    now = time.time()
    if _KNOWN_TAGS_CACHE and now - _KNOWN_TAGS_FETCHED_AT < 300:
        return _KNOWN_TAGS_CACHE

    try:
        result = await db_session.execute(text("SELECT tag FROM tag_stats"))
        rows = result.fetchall()
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
    limit: int = 50,
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
        params: Dict[str, object] = {"limit": limit}
        joins: List[str] = []
        where_clauses: List[str] = []
        cte_segment: Optional[str] = None

        # doujinshi only
        where_clauses.append("g.manga_type = 'doujinshi'")

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
        sql_segments.append("LIMIT :limit")
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
            limit=limit,
            exclude_tag=exclude_tag,
        )
async def search_galleries(
    db_session: AsyncSession,
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
        Gallery.page_count,
        Gallery.created_at_unix,
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
                select(PageView)
                .where(PageView.session_id == session_id)
                .order_by(PageView.id.desc())
                .limit(max_page_views)
            )
            result = await tracking_db.execute(stmt)
            page_views = result.scalars().all()
    except Exception as exc:
        logger.error("セッションプロファイル取得エラー: %s", exc)
        return {}

    if not page_views:
        return {}

    from datetime import timezone
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
        logger.error("ギャラリータグ取得エラー: %s", exc)
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


# =================================================================================
# vvvvvvvvvvvvvvvvvvvvvvvv 高速化のための修正箇所 vvvvvvvvvvvvvvvvvvvvvvvv
# =================================================================================
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
            except:
                return True
        sorted_galleries = [g for g in sorted_galleries if check_exclude(g)]

    galleries = sorted_galleries[:limit]
    final_gallery_ids = {g["gallery_id"] for g in galleries}
    exclude_ids.update(final_gallery_ids)

    # ステップ3: 結果が不足している場合、最新のギャラリーで補完
    if len(galleries) < limit:
        remaining = limit - len(galleries)
        fallback_params: Dict[str, Any] = {"limit": remaining}
        fallback_clauses = ["g.manga_type = 'doujinshi'"]

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

# =================================================================================
# ^^^^^^^^^^^^^^^^^^^^^^^^ 高速化のための修正箇所 ^^^^^^^^^^^^^^^^^^^^^^^^^
# =================================================================================

def _derive_filename(url: str) -> str:
    trimmed = url.split("?", 1)[0].rstrip("/")
    candidate = trimmed.split("/")[-1] if trimmed else ""
    return candidate or "download.bin"


_IMAGE_RESOLVER_FAILURE_AT: float = 0.0
_IMAGE_RESOLVER_FAILURE_COOLDOWN = 120.0
_IMAGE_RESOLVER_TIMEOUT = 3.0


async def _ensure_image_resolver_ready() -> bool:
    """Initialise the image resolver without blocking the event loop."""

    global _IMAGE_RESOLVER_FAILURE_AT

    now = time.monotonic()
    if _IMAGE_RESOLVER_FAILURE_AT and now - _IMAGE_RESOLVER_FAILURE_AT < _IMAGE_RESOLVER_FAILURE_COOLDOWN:
        return False

    try:
        await asyncio.wait_for(ImageUriResolver.async_synchronize(), timeout=_IMAGE_RESOLVER_TIMEOUT)
        return True
    except asyncio.TimeoutError:
        logger.warning("ImageUriResolver 同期がタイムアウトしました")
    except Exception as exc:
        logger.error("ImageUriResolver 初期化エラー: %s", exc)

    _IMAGE_RESOLVER_FAILURE_AT = now
    return False


async def geturl(gi: Dict[str, Any]) -> List[str]:
    files = gi.get("files", []) or []
    if not files:
        return []

    resolver_ready = await _ensure_image_resolver_ready()
    if not resolver_ready:
        return []

    urls: List[str] = []
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

class TagTranslationsUpdateRequest(BaseModel):
    translations: Dict[str, str] = Field(default_factory=dict)


class TagCategoryModel(BaseModel):
    id: str
    label: str
    tags: List[str] = Field(default_factory=list)


class TagCategoriesUpdateRequest(BaseModel):
    categories: List[TagCategoryModel] = Field(default_factory=list)

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

@app.get("/history", response_class=HTMLResponse)
async def read_history():
    return _serve_cached_html("template/history.html")


@app.get("/tag-editor", response_class=HTMLResponse)
async def read_tag_editor():
    return _serve_cached_html("template/tag-editor.html")

@app.post("/search")
async def search_galleries_endpoint(request: SearchRequest):
    try:
        async with get_db_session() as db:
            results = await search_galleries_fast(
                db,
                title=request.title,
                tag=request.tag,
                exclude_tag=request.exclude_tag,
                character=request.character,
                limit=request.limit,
                after_created_at=request.after_created_at,
                after_gallery_id=request.after_gallery_id,
                min_pages=request.min_pages,
                max_pages=request.max_pages,
            )

            for result in results:
                try:
                    files_data = json.loads(result["files"]) if isinstance(result.get("files"), str) else result.get("files")
                    files_list = files_data if isinstance(files_data, list) else []
                    gallery_info = {"gallery_id": result["gallery_id"], "files": files_data}
                    result["image_urls"] = await geturl(gallery_info)
                    stored_pages = result.get("page_count")
                    if not isinstance(stored_pages, int) or stored_pages < 0:
                        result["page_count"] = len(files_list)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"files 解析エラー: {e}, gallery_id: {result['gallery_id']}")
                    result["image_urls"] = []
                    result["page_count"] = 0
                if "files" in result:
                    del result["files"]

            # Derive the next cursor from the final item in this page
            next_after_created_at = None
            next_after_gallery_id = None
            if results and len(results) == request.limit:
                last_item = results[-1]
                next_after_created_at = last_item.get("created_at")
                next_after_gallery_id = last_item.get("gallery_id")

            return {
                "results": results,
                "count": len(results),
                "has_more": len(results) == request.limit,
                "next_after_created_at": next_after_created_at,
                "next_after_gallery_id": next_after_gallery_id,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"検索エラー: {str(e)}")


# =================================================================================
# vvvvvvvvvvvvvvvvvvvvvvvv 高速化のための修正箇所 vvvvvvvvvvvvvvvvvvvvvvvv
# =================================================================================
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
                    logger.error("おすすめ個人化プロファイル作成エラー: %s", exc)
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
            for result in results:
                try:
                    files_data = json.loads(result.get("files")) if isinstance(result.get("files"), str) else result.get("files")
                except (json.JSONDecodeError, TypeError):
                    files_data = []
                files_list = files_data if isinstance(files_data, list) else []
                gallery_info = {"gallery_id": result["gallery_id"], "files": files_list}
                geturl_tasks.append(geturl(gallery_info))
            
            # asyncio.gatherで全てのgeturlコルーチンを並列実行
            image_urls_results = await asyncio.gather(*geturl_tasks, return_exceptions=True)

            # 3. 結果をマージして最終的なペイロードを生成
            payload: List[Dict[str, Any]] = []
            for i, result in enumerate(results):
                image_urls = image_urls_results[i]
                if isinstance(image_urls, Exception):
                    logger.error(f"geturlがギャラリーID {result['gallery_id']} で失敗: {image_urls}")
                    image_urls = []

                # page_countを再計算または検証
                try:
                    files_data = json.loads(result.get("files")) if isinstance(result.get("files"), str) else result.get("files")
                    files_list = files_data if isinstance(files_data, list) else []
                except (json.JSONDecodeError, TypeError):
                    files_list = []
                
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
        logger.exception("おすすめ取得APIで予期せぬエラーが発生しました")
        raise HTTPException(status_code=500, detail=f"おすすめ取得エラー: {str(e)}")

# =================================================================================
# ^^^^^^^^^^^^^^^^^^^^^^^^ 高速化のための修正箇所 ^^^^^^^^^^^^^^^^^^^^^^^^^
# =================================================================================

@app.get("/search")
async def search_galleries_get(
    title: Optional[str] = None,
    tag: Optional[str] = None,
    exclude_tag: Optional[str] = None,
    character: Optional[str] = None,
    limit: int = 50,
    after_created_at: Optional[str] = None,
    after_gallery_id: Optional[int] = None,
    min_pages: Optional[int] = None,
    max_pages: Optional[int] = None,
    sort_by: Optional[str] = None,
):
    try:
        async with get_db_session() as db:
            # sort_byパラメータがランキングの場合は特別処理
            if sort_by and sort_by in ['daily', 'weekly', 'monthly', 'yearly', 'all_time']:
                # ランキングファイルからIDを読み込む
                ranking_ids = await _load_ranking_ids(sort_by)
                
                # タグ検索の場合はタグでフィルタリング
                if tag:
                    known_tags = await _get_known_tag_set(db)
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
                
                # ランキング順にギャラリー情報を取得
                if filtered_ranking_ids:
                    # リミットとオフセットを適用
                    start_idx = 0  # 簡略化のため常に先頭から
                    end_idx = min(start_idx + limit, len(filtered_ranking_ids))
                    paginated_ids = filtered_ranking_ids[start_idx:end_idx]
                    
                    placeholders = ', '.join([f':id_{i}' for i in range(len(paginated_ids))])
                    params = {f'id_{i}': gallery_id for i, gallery_id in enumerate(paginated_ids)}
                    
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
                        ORDER BY g.gallery_id DESC
                    """
                    
                    result = await db.execute(text(query), params)
                    results = [_serialize_gallery(row) for row in result.mappings()]
                else:
                    results = []
            else:
                # 通常の検索
                results = await search_galleries_fast(
                    db,
                    title=title,
                    tag=tag,
                    exclude_tag=exclude_tag,
                    character=character,
                    limit=limit,
                    after_created_at=after_created_at,
                    after_gallery_id=after_gallery_id,
                    min_pages=min_pages,
                    max_pages=max_pages,
                )

            for result in results:
                try:
                    files_data = json.loads(result["files"]) if isinstance(result.get("files"), str) else result.get("files")
                    files_list = files_data if isinstance(files_data, list) else []
                    gallery_info = {"gallery_id": result["gallery_id"], "files": files_data}
                    result["image_urls"] = await geturl(gallery_info)
                    stored_pages = result.get("page_count")
                    if not isinstance(stored_pages, int) or stored_pages < 0:
                        result["page_count"] = len(files_list)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"files 解析エラー: {e}, gallery_id: {result['gallery_id']}")
                    result["image_urls"] = []
                    result["page_count"] = 0
                if "files" in result:
                    del result["files"]

            # Derive the next cursor from the final item in this page
            next_after_created_at = None
            next_after_gallery_id = None
            if results and len(results) == limit:
                last_item = results[-1]
                next_after_created_at = last_item.get("created_at")
                next_after_gallery_id = last_item.get("gallery_id")

            return {
                "results": results,
                "count": len(results),
                "has_more": len(results) == limit,
                "next_after_created_at": next_after_created_at,
                "next_after_gallery_id": next_after_gallery_id,
            }
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
                    if resp.status == 404:
                        # 404エラーの場合はImageUriResolverを強制同期して再試行
                        logger.info("404エラーを検出、ImageUriResolverを再同期します")
                        try:
                            await ImageUriResolver.async_synchronize(force=True)
                            logger.info("ImageUriResolverの再同期が完了しました。再試行します")
                        except Exception as e:
                            logger.error(f"ImageUriResolver再同期エラー: {e}")
                        if attempt < max_retries:
                            await asyncio.sleep(retry_delay * (2 ** attempt))
                            continue
                        raise HTTPException(status_code=resp.status, detail=f"Upstream 4xx: {resp.status}")
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
                "page_count": gallery.page_count if isinstance(gallery.page_count, int) and gallery.page_count >= 0 else len(files_list),
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
        async with get_db_session() as db:
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

            total_result = await db.execute(text(total_sql), params)
            total_count = total_result.scalar()
            rows_result = await db.execute(text(query), params)
            rows = rows_result.fetchall()
            tags = [{"tag": r.tag, "count": r.count} for r in rows]
            return {"tags": tags, "total": total_count, "has_more": (offset + limit) < (total_count or 0)}
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
    data = await _read_json_file(TAG_TRANSLATIONS_FILE, {})
    if not isinstance(data, Mapping):
        raise HTTPException(status_code=500, detail="タグ翻訳データの形式が正しくありません")

    translations: Dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        translations[key] = value if isinstance(value, str) else ""
    return {"translations": translations}


@app.post("/api/tag-translations")
async def update_tag_translations(request: TagTranslationsUpdateRequest) -> Dict[str, Any]:
    processed: Dict[str, str] = {}
    seen: Set[str] = set()

    for raw_key, raw_value in request.translations.items():
        tag = (raw_key or "").strip()
        if not tag:
            continue
        normalised = tag.lower()
        if normalised in seen:
            raise HTTPException(status_code=400, detail=f"タグが重複しています: {tag}")
        seen.add(normalised)
        translation = (raw_value or "").strip()
        processed[tag] = translation

    await _write_json_file(TAG_TRANSLATIONS_FILE, processed, sort_keys=True)
    return {"status": "ok", "count": len(processed)}


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
# トラッキング系
# =========================
@app.post("/api/tracking/session")
async def update_session(request: SessionRequest):
    try:
        async with get_tracking_db_session() as db:
            stmt = select(UserSession).where(UserSession.session_id == request.session_id)
            result = await db.execute(stmt)
            existing_session = result.scalars().first()
            now_iso = datetime.now().isoformat()
            if existing_session:
                existing_session.last_activity = now_iso
                if request.user_agent:
                    existing_session.user_agent = request.user_agent
                if request.ip_address:
                    existing_session.ip_address = request.ip_address
                await db.commit()
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
                await db.commit()
                return {"status": "created", "session_id": new_session.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"セッション更新エラー: {str(e)}")

@app.post("/api/tracking/page-view")
async def record_page_view(request: PageViewRequest):
    try:
        async with get_tracking_db_session() as db:
            stmt = select(UserSession).where(UserSession.session_id == request.session_id)
            result = await db.execute(stmt)
            session = result.scalars().first()
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
            await db.commit()
            return {"status": "created", "page_view_id": new_page_view.id, "session_id": request.session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ページビュー記録エラー: {str(e)}")

@app.post("/api/tracking/events")
async def record_events(request: BatchEventsRequest):
    try:
        async with get_tracking_db_session() as db:
            session_ids = {event.session_id for event in request.events}
            sessions: List[UserSession]
            if session_ids:
                stmt = select(UserSession).where(UserSession.session_id.in_(session_ids))
                result = await db.execute(stmt)
                sessions = result.scalars().all()
            else:
                sessions = []
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
            await db.commit()
            return {"status": "created", "count": len(new_events), "session_id": valid_events[0].session_id if valid_events else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"イベント記録エラー: {str(e)}")

@app.post("/api/tracking/single-event")
async def record_single_event(request: EventRequest):
    try:
        async with get_tracking_db_session() as db:
            stmt = select(UserSession).where(UserSession.session_id == request.session_id)
            result = await db.execute(stmt)
            session = result.scalars().first()
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
            await db.commit()
            return {"status": "created", "event_id": new_event.id, "session_id": request.session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"イベント記録エラー: {str(e)}")

# =========================
# ランキング系API
# =========================
@app.get("/api/rankings")
async def get_rankings(
    ranking_type: str = "daily",
    limit: int = 50,
    offset: int = 0,
    tag: Optional[str] = None
):
    """
    ランキングデータを取得するAPI
    ranking_type: 'daily', 'weekly', 'monthly', 'yearly', 'all_time'
    tag: タグでフィルタリング（オプション）
    """
    try:
        # ファイルからランキングIDを読み込む
        ranking_ids = await _load_ranking_ids(ranking_type)
        
        # オフセットとリミットを適用
        start_index = offset
        end_index = offset + limit
        paginated_ids = ranking_ids[start_index:end_index]
        
        if not paginated_ids:
            return {
                "rankings": [],
                "count": 0,
                "total": len(ranking_ids),
                "has_more": False
            }
        
        # データベースからギャラリー情報を取得
        async with get_db_session() as db:
            placeholders = ', '.join([f':id_{i}' for i in range(len(paginated_ids))])
            params = {f'id_{i}': gallery_id for i, gallery_id in enumerate(paginated_ids)}
            
            # タグフィルタリング用のWHERE句を構築
            tag_filter_clause = ""
            if tag:
                known_tags = await _get_known_tag_set(db)
                tag_terms = _parse_tag_terms(tag, known_tags)
                if tag_terms:
                    # EXISTS句を使用してタグAND検索を実装
                    tag_exists_clauses, tag_params = _build_tag_exists_clause("g", tag_terms)
                    params.update(tag_params)
                    tag_filter_clause = f"AND {' AND '.join(tag_exists_clauses)}"
            
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
                WHERE g.gallery_id IN ({placeholders}) {tag_filter_clause}
                ORDER BY g.gallery_id DESC
            """
            
            result = await db.execute(text(query), params)
            rows = result.fetchall()
            
            # ギャラリーIDをキーとして行をマッピング
            gallery_dict = {row.gallery_id: row for row in rows}
            
            rankings = []
            for rank_position, gallery_id in enumerate(paginated_ids, start=offset + 1):
                if gallery_id in gallery_dict:
                    row = gallery_dict[gallery_id]
                    ranking_data = {
                        "gallery_id": row.gallery_id,
                        "ranking_type": ranking_type,
                        "rank": rank_position,
                        "japanese_title": row.japanese_title,
                        "tags": row.tags,
                        "characters": row.characters,
                        "page_count": row.page_count,
                        "created_at": row.created_at,
                        "created_at_unix": row.created_at_unix
                    }
                    
                    # 画像URLを取得
                    try:
                        files_data = json.loads(row.files) if hasattr(row, 'files') and row.files else []
                        files_list = files_data if isinstance(files_data, list) else []
                        gallery_info = {"gallery_id": row.gallery_id, "files": files_list}
                        ranking_data["image_urls"] = await geturl(gallery_info)
                    except (json.JSONDecodeError, TypeError, AttributeError) as e:
                        logger.error(f"ランキングデータの画像URL取得エラー: {e}, gallery_id: {row.gallery_id}")
                        ranking_data["image_urls"] = []
                    
                    rankings.append(ranking_data)
            
            return {
                "rankings": rankings,
                "count": len(rankings),
                "total": len(ranking_ids),
                "has_more": (offset + limit) < len(ranking_ids)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ランキングデータの取得エラー: {str(e)}")

@app.post("/api/rankings/update")
async def update_rankings():
    """
    ランキングデータを更新するAPI（現在は無効化）
    *_ids.txtファイルから直接読み込むため、このエンドポイントは不要
    """
    return {"status": "disabled", "message": "ランキング更新は無効化されています。*_ids.txtファイルから直接読み込まれます。"}

# ランキング更新関数は不要になったため削除
# *_ids.txtファイルから直接読み込むため

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
            logger.info(f"次回のランキングファイル更新は {next_update.strftime('%Y-%m-%d %H:%M:%S')} に実行")
            await asyncio.sleep(wait_seconds)
            
            logger.info("ランキングファイル更新開始")
            try:
                # hitomi.pyのdownload_all_popular_files()を実行
                import sys
                import subprocess
                
                # スクリプトを実行
                result = subprocess.run(
                    [sys.executable, "scraper/hitomi.py"],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5分タイムアウト
                )
                
                if result.returncode == 0:
                    logger.info("ランキングファイル更新完了")
                    logger.info(f"出力: {result.stdout}")
                else:
                    logger.error(f"ランキングファイル更新エラー: {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.error("ランキングファイル更新がタイムアウトしました")
            except Exception as e:
                logger.error(f"ランキングファイル更新中にエラー: {str(e)}")
        except Exception as e:
            logger.error(f"ランキング更新タスク全体でエラー: {str(e)}")
            await asyncio.sleep(3600)  # エラーの場合は1時間待って再試行

# =========================
# ライフサイクル
# =========================
@app.on_event("startup")
async def startup_event():
    global global_session, scheduler_task

    # DB 初期化
    await init_database()
    logger.info("メインDB初期化完了")
    await init_tracking_database()
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
    
    # ランキング更新スケジューラ開始
    ranking_scheduler_task = asyncio.create_task(daily_ranking_update_task())
    logger.info("ランキングファイル更新スケジューラ開始")

@app.on_event("shutdown")
async def shutdown_event():
    global global_session, scheduler_task, ranking_scheduler_task
    if 'scheduler_task' in globals() and scheduler_task:
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            logger.info("同期スケジューラ停止")
    if 'ranking_scheduler_task' in globals() and ranking_scheduler_task:
        ranking_scheduler_task.cancel()
        try:
            await ranking_scheduler_task
        except asyncio.CancelledError:
            logger.info("ランキング更新スケジューラ停止")
    if global_session:
        await global_session.close()
        logger.info("Global session closed")

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
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log)
