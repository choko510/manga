import pytest
import pytest_asyncio
import asyncio
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

# Import main after setting up environment if needed, but here we just import it
# CAUTION: This imports main.py which executes global code. 
# Ideally main.py should not have side effects on import, but it does (creating engine).
from main import app, get_db_session, Base, global_state, init_database

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="function")
async def test_engine():
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    yield engine
    await engine.dispose()

@pytest_asyncio.fixture(scope="function")
async def test_db(test_engine):
    # Patch main.engine so init_database uses our test engine
    with patch("main.engine", test_engine):
        # Initialize schema (tables, FTS, triggers)
        await init_database()

    TestingSessionLocal = async_sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    
    async with TestingSessionLocal() as session:
        yield session


@pytest.fixture(scope="function", autouse=True)
def mock_background_tasks():
    """Prevent background tasks and real DB initialization from running during tests."""
    with patch("main.init_database", new_callable=MagicMock) as mock_init, \
         patch("main.hourly_sync_task", new_callable=MagicMock) as mock_sync, \
         patch("main.daily_ranking_update_task", new_callable=MagicMock) as mock_ranking, \
         patch("main._precache_search_counts", new_callable=MagicMock) as mock_precache, \
         patch("main._warmup_connections", new_callable=MagicMock) as mock_warmup:
        yield

@pytest_asyncio.fixture(scope="function")
async def client(test_engine):
    TestingSessionLocal = async_sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )

    # Patch get_db_session because endpoints call it directly
    with patch("main.get_db_session", side_effect=TestingSessionLocal):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
