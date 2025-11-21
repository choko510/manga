import os
import asyncio
import json
from typing import Set, List
from datetime import datetime, timezone
from sqlalchemy import create_engine, text, Column, Integer, String, Text, Computed
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# プロジェクトのモジュールをインポート
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.gallery import get_gallery
from lib.types import Gallery as GalleryType

# データベースベースクラス
Base = declarative_base()

# Galleryモデル（main.pyからコピー）
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
    created_at_unix = Column(
        Integer,
        Computed("CAST(strftime('%s', created_at) AS INTEGER)", persisted=True),
    )

    def __repr__(self):
        return f"<Gallery(id={self.gallery_id}, title='{self.japanese_title}')>"

def load_ids_from_file(filename: str) -> Set[int]:
    """
    ファイルからIDリストを読み込む
    
    Args:
        filename: IDが保存されているファイルパス
    
    Returns:
        IDのセット
    """
    ids = set()
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    ids.add(int(line))
        print(f"{filename}から {len(ids)}件のIDを読み込みました")
    except FileNotFoundError:
        print(f"{filename}が見つかりません")
    except Exception as e:
        print(f"{filename}の読み込み中にエラーが発生しました: {e}")
    
    return ids

def get_missing_ids() -> Set[int]:
    """
    galleries.txtになくてall_ids.txtにあるIDを取得する
    
    Returns:
        差分IDのセット
    """
    # 既存のギャラリーIDを読み込む
    existing_ids = load_ids_from_file("galleries.txt")
    
    # 全てのIDを読み込む
    all_ids = load_ids_from_file("all_ids.txt")
    
    # 差分を計算（all_idsにあってgalleries.txtにないID）
    missing_ids = all_ids - existing_ids
    
    print(f"追加が必要なギャラリー数: {len(missing_ids)}")
    
    return missing_ids

async def fetch_gallery_data(gallery_id: int) -> GalleryType:
    """
    hitomi.laからギャラリーデータを取得
    
    Args:
        gallery_id: ギャラリーID
    
    Returns:
        ギャラリーデータ
    """
    try:
        gallery = get_gallery(gallery_id)
        return gallery
    except Exception as e:
        print(f"ギャラリー{gallery_id}の取得中にエラーが発生しました: {e}")
        raise

def gallery_to_db_model(gallery: GalleryType) -> Gallery:
    """
    GalleryTypeをデータベースモデルに変換
    
    Args:
        gallery: ギャラリーデータ
    
    Returns:
        データベースモデル
    """
    # タグをJSON文字列に変換
    tags_json = json.dumps([
        {"type": tag.type, "name": tag.name} 
        for tag in gallery.tags
    ], ensure_ascii=False)
    
    # ファイル情報をJSON文字列に変換
    files_json = json.dumps([
        {
            "hash": img.hash,
            "has_avif": img.has_avif,
            "has_webp": img.has_webp,
            "has_jxl": img.has_jxl
        } 
        for img in gallery.files
    ], ensure_ascii=False)
    
    # キャラクター情報をJSON文字列に変換
    characters_json = json.dumps(gallery.characters, ensure_ascii=False)
    
    # 作成日時をISO8601形式に変換
    created_at = gallery.published_date.isoformat() if gallery.published_date else datetime.now(timezone.utc).isoformat()
    
    return Gallery(
        gallery_id=gallery.id,
        japanese_title=gallery.title.japanese or gallery.title.display,
        tags=tags_json,
        characters=characters_json,
        files=files_json,
        manga_type=gallery.type,
        created_at=created_at,
        page_count=len(gallery.files) if gallery.files else None
    )

async def save_gallery_to_db(session: AsyncSession, gallery: GalleryType) -> bool:
    """
    ギャラリーデータをデータベースに保存
    
    Args:
        session: データベースセッション
        gallery: ギャラリーデータ
    
    Returns:
        保存成功の場合True
    """
    try:
        # データベースモデルに変換
        db_gallery = gallery_to_db_model(gallery)
        
        # データベースに保存
        session.add(db_gallery)
        await session.commit()
        
        print(f"ギャラリー{gallery.id}を保存しました: {db_gallery.japanese_title}")
        return True
        
    except Exception as e:
        await session.rollback()
        print(f"ギャラリー{gallery.id}の保存中にエラーが発生しました: {e}")
        return False

async def process_missing_galleries(missing_ids: Set[int], batch_size: int = 10) -> None:
    """
    欠けているギャラリーを処理
    
    Args:
        missing_ids: 追加が必要なギャラリーIDのセット
        batch_size: 一度に処理するバッチサイズ
    """
    # 非同期データベースエンジンを作成
    engine = create_async_engine(
        "sqlite+aiosqlite:///db/sa.db",
        echo=False,
        connect_args={"timeout": 20}
    )
    
    # セッションファクトリを作成
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    # IDをソートして処理
    sorted_ids = sorted(missing_ids)
    total_count = len(sorted_ids)
    processed_count = 0
    success_count = 0
    
    print(f"処理を開始します。合計{total_count}件のギャラリーを追加します。")
    
    # バッチ処理
    for i in range(0, total_count, batch_size):
        batch_ids = sorted_ids[i:i + batch_size]
        
        async with async_session() as session:
            for gallery_id in batch_ids:
                try:
                    processed_count += 1
                    print(f"[{processed_count}/{total_count}] ギャラリー{gallery_id}を処理中...")
                    
                    # ギャラリーデータを取得
                    gallery = await fetch_gallery_data(gallery_id)
                    
                    # データベースに保存
                    if await save_gallery_to_db(session, gallery):
                        success_count += 1
                    
                    # 少し待機してサーバーに負荷をかけない
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    print(f"ギャラリー{gallery_id}の処理中にエラーが発生しました: {e}")
                    continue
        
        # バッチ完了時に進捗を表示
        batch_end = min(i + batch_size, total_count)
        print(f"バッチ完了: {batch_end}/{total_count} (成功: {success_count})")
        
        # バッチ間で少し待機
        if i + batch_size < total_count:
            await asyncio.sleep(2)
    
    print(f"処理完了！成功: {success_count}/{total_count}")

async def main():
    """
    メイン処理
    """
    print("=== 欠けているギャラリーの同期を開始 ===")
    
    # 1. 差分を取得
    missing_ids = get_missing_ids()
    
    if not missing_ids:
        print("追加が必要なギャラリーはありません。")
        return
    
    # 2. 欠けているギャラリーを処理
    await process_missing_galleries(missing_ids)
    
    print("=== 同期完了 ===")

if __name__ == "__main__":
    asyncio.run(main())