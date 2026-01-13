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
    # Setup - add dummy gallery to DB
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

    # Verify DB content
    res = await test_db.execute(text("SELECT COUNT(*) FROM galleries"))
    count = res.scalar()
    assert count == 1, "DB setup failed: galleries table empty"
    
    # Test list all galleries (no params)
    response = await client.get("/search")
    assert response.status_code == 200, f"Search all failed: {response.text}"
    data_all = response.json()
    assert data_all["count"] == 1, f"Expected 1 gallery, got {data_all['count']}"

    # Test basic search by title
    response = await client.get("/search", params={"title": "UniqueTitleXYZ"})
    assert response.status_code == 200, f"Search by title failed: {response.text}"
    data = response.json()
    assert data["count"] == 1, "Expected 1 gallery match for title"
    assert data["results"][0]["japanese_title"] == "UniqueTitleXYZ"

    # Test tag search (mocking known_tags in main if necessary, or assuming they work)
    # Note: Tag search relies on gallery_tags table which is populated triggers.
    response = await client.get("/search", params={"tag": "test_tag"})
    assert response.status_code == 200
    data_tag = response.json()
    # Tag search logic involves intersection with ranking if sort_by or similar? 
    # Or just search_galleries_fast.
    # It should work ideally.
