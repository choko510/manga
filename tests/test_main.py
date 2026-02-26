import pytest
from httpx import AsyncClient, ASGITransport
import json
from main import app, Gallery
from sqlalchemy import text

@pytest.mark.asyncio
async def test_read_index(client: AsyncClient):
    response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

@pytest.mark.asyncio
async def test_search_galleries(client: AsyncClient, test_db):
    # セットアップ - テスト用ギャラリーをDBに追加
    gallery = Gallery(
        japanese_title="UniqueTitleXYZ",
        tags=json.dumps(["test_tag", "demo"]),
        characters="test_char",
        files="[]",
        manga_type="doujinshi",
        created_at="2024-01-01T00:00:00Z",
        page_count=10,
        created_at_unix=1704067200
    )
    test_db.add(gallery)
    await test_db.commit()
    await test_db.refresh(gallery)

    # DBの内容を検証
    res = await test_db.execute(text("SELECT COUNT(*) FROM galleries"))
    count = res.scalar()
    assert count == 1, "DBセットアップ失敗: galleriesテーブルが空です"
    
    # 全ギャラリーの取得テスト (パラメータなし)
    response = await client.get("/search")
    assert response.status_code == 200, f"全件検索失敗: {response.text}"
    data_all = response.json()
    assert data_all["count"] == 1, f"1件のギャラリーを期待していましたが、{data_all['count']}件取得されました"

    # タイトルによる基本検索のテスト
    response = await client.get("/search", params={"title": "UniqueTitleXYZ"})
    assert response.status_code == 200, f"タイトル検索失敗: {response.text}"
    data = response.json()
    assert data["count"] == 1, "タイトル一致で1件のギャラリーを期待していました"
    assert data["results"][0]["japanese_title"] == "UniqueTitleXYZ"

    # タグ検索のテスト（必要に応じて main の known_tags をモックするか、正常に動作すると想定）
    # 注: タグ検索はトリガーによってデータが投入される gallery_tags テーブルに依存します。
    response = await client.get("/search", params={"tag": "test_tag"})
    assert response.status_code == 200
    data_tag = response.json()
    # タグ検索のロジックには、sort_by などによるランキングとの交差が含まれるか？
    # または単に search_galleries_fast。
    # 本来は正常に動作するはずです。
