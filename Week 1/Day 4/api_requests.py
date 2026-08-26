"""
Week 1 | Day 4 — API Requests & HTTP Basics
============================================
Internship at Al Aziz Technologies | AI Engineering Track

Topics covered:
    - HTTP verbs: GET, POST (conceptual)
    - Status codes and response parsing
    - requests.get() with timeout, headers, params
    - Graceful error handling (HTTPError, ConnectionError, Timeout)
    - Consuming a real public API: Open-Meteo (free, no API key required)
    - Consuming a real public API: JSONPlaceholder (REST test API)

Run:
    python "Week 1/Day 4/api_requests.py"
    (Requires internet access; gracefully degrades if offline.)
"""

import logging
import sys
import time
from typing import Any

try:
    import requests
    from requests.exceptions import ConnectionError, HTTPError, Timeout
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("day4.api")

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_TIMEOUT: int = 10   # seconds
ISLAMABAD_LAT: float = 33.6844
ISLAMABAD_LON: float = 73.0479


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def safe_get(url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Perform a GET request with full error handling.

    Args:
        url: Endpoint URL.
        params: Optional query parameters.

    Returns:
        Parsed JSON dict, or None on any failure.
    """
    if not REQUESTS_AVAILABLE:
        logger.error("'requests' library not installed. Run: pip install requests")
        return None

    headers = {"Accept": "application/json", "User-Agent": "AlAziz-Internship/1.0"}
    logger.info("GET %s  params=%s", url, params)
    start = time.perf_counter()
    try:
        response = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
        elapsed = time.perf_counter() - start
        logger.info("Response: HTTP %d  (%.2fs)", response.status_code, elapsed)
        response.raise_for_status()
        return response.json()
    except HTTPError as exc:
        logger.error("HTTP error %s — %s", exc.response.status_code, exc)
    except ConnectionError:
        logger.error("Connection failed — no internet or DNS error.")
    except Timeout:
        logger.error("Request timed out after %ds.", DEFAULT_TIMEOUT)
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 1. OPEN-METEO — free weather API (no key required)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_weather(lat: float = ISLAMABAD_LAT, lon: float = ISLAMABAD_LON) -> None:
    """Fetch current weather for Islamabad from Open-Meteo."""
    logger.info("── Open-Meteo Weather API ──")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        "wind_speed_unit": "kmh",
        "timezone": "Asia/Karachi",
    }

    data = safe_get(url, params=params)
    if data is None:
        print("\n  ✗ Weather data unavailable (check internet connection).")
        return

    current: dict[str, Any] = data.get("current", {})
    units: dict[str, str] = data.get("current_units", {})

    wmo_codes: dict[int, str] = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 51: "Light drizzle", 61: "Slight rain", 71: "Slight snow",
        80: "Slight showers", 95: "Thunderstorm",
    }
    code: int = current.get("weather_code", -1)
    condition: str = wmo_codes.get(code, f"WMO code {code}")

    print("\n── Weather: Islamabad, Pakistan (Open-Meteo) ──")
    print(f"  Temperature  : {current.get('temperature_2m')} {units.get('temperature_2m', '°C')}")
    print(f"  Humidity     : {current.get('relative_humidity_2m')} {units.get('relative_humidity_2m', '%')}")
    print(f"  Wind Speed   : {current.get('wind_speed_10m')} {units.get('wind_speed_10m', 'km/h')}")
    print(f"  Condition    : {condition}")
    print(f"  Time         : {current.get('time', 'N/A')}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. JSONPLACEHOLDER — REST test API
# ══════════════════════════════════════════════════════════════════════════════
def fetch_posts(limit: int = 5) -> None:
    """Fetch and display sample blog posts from JSONPlaceholder."""
    logger.info("── JSONPlaceholder Posts API ──")
    url = "https://jsonplaceholder.typicode.com/posts"

    data = safe_get(url)
    if data is None:
        print("\n  ✗ Posts data unavailable.")
        return

    posts: list[dict[str, Any]] = data[:limit]
    print(f"\n── JSONPlaceholder: Top {limit} Posts ──")
    for post in posts:
        title = post.get("title", "")[:55]
        body_preview = post.get("body", "").replace("\n", " ")[:60]
        print(f"  [{post['id']:>2}] {title}")
        print(f"       {body_preview}…\n")


def fetch_user(user_id: int = 1) -> None:
    """Fetch a single user record from JSONPlaceholder."""
    logger.info("── JSONPlaceholder User API (id=%d) ──", user_id)
    url = f"https://jsonplaceholder.typicode.com/users/{user_id}"

    data = safe_get(url)
    if data is None:
        print(f"\n  ✗ User {user_id} data unavailable.")
        return

    print(f"\n── User #{user_id} Profile ──")
    print(f"  Name     : {data.get('name')}")
    print(f"  Username : {data.get('username')}")
    print(f"  Email    : {data.get('email')}")
    company: dict = data.get("company", {})
    print(f"  Company  : {company.get('name')} — {company.get('catchPhrase')}")
    address: dict = data.get("address", {})
    print(f"  City     : {address.get('city')}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. HTTP CONCEPTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def explain_http_basics() -> None:
    """Print a concise HTTP reference card."""
    print("\n" + "=" * 60)
    print("  HTTP BASICS — Reference Card")
    print("=" * 60)

    verbs = [
        ("GET",    "Retrieve a resource (read-only, cacheable)"),
        ("POST",   "Create a new resource (send body)"),
        ("PUT",    "Replace an existing resource"),
        ("PATCH",  "Partially update a resource"),
        ("DELETE", "Remove a resource"),
    ]
    print("\n  ── HTTP Verbs ──")
    for verb, desc in verbs:
        print(f"  {verb:<7} — {desc}")

    codes = [
        (200, "OK"),
        (201, "Created"),
        (400, "Bad Request"),
        (401, "Unauthorized"),
        (403, "Forbidden"),
        (404, "Not Found"),
        (422, "Unprocessable Entity"),
        (429, "Too Many Requests"),
        (500, "Internal Server Error"),
        (503, "Service Unavailable"),
    ]
    print("\n  ── Common Status Codes ──")
    for code, meaning in codes:
        symbol = "✓" if code < 400 else "✗"
        print(f"  {symbol} {code} — {meaning}")

    print("\n  ── Anatomy of a requests call ──")
    print("""
  import requests

  response = requests.get(
      "https://api.example.com/data",
      params  = {"limit": 10, "format": "json"},
      headers = {"Authorization": "Bearer YOUR_KEY"},
      timeout = 10,                      # always set a timeout!
  )
  response.raise_for_status()            # raises HTTPError on 4xx/5xx
  data = response.json()                 # parse JSON body
""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "█" * 60)
    print("  WEEK 1 | DAY 4 — API Requests & HTTP Basics")
    print("  Internship at Al Aziz Technologies")
    print("█" * 60)

    explain_http_basics()
    fetch_weather()
    fetch_posts(limit=3)
    fetch_user(user_id=2)

    print("\n" + "=" * 60)
    logger.info("Day 4 — API Requests complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical("Unhandled exception: %s", exc, exc_info=True)
        sys.exit(1)
