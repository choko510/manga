import aiohttp

YearURL = "https://ltn.gold-usergeneratedcontent.net/popular/year-all.nozomi"
MonthURL = "https://ltn.gold-usergeneratedcontent.net/popular/month-all.nozomi"
WeekURL = "https://ltn.gold-usergeneratedcontent.net/popular/week-all.nozomi"
DayURL = "https://ltn.gold-usergeneratedcontent.net/popular/day-all.nozomi"

async def fetch_hitomi_data(session: aiohttp.ClientSession, url: str) -> dict:
    async with session.get(url) as response:
        return await response.json()



