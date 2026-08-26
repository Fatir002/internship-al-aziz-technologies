# 🌐 Week 1 | Day 4 — REST API Consumption & HTTP Basics

> **Internship at Al Aziz Technologies · AI Engineering Track**

Practical, production-style API integration using Python's `requests` library — consuming two real public APIs with proper timeout handling, structured error catching, and response parsing. Includes a compact HTTP reference card covering verbs, status codes, and request anatomy.

---

## 📋 Topics Covered

| # | Topic |
|---|-------|
| 1 | HTTP verbs — GET, POST, PUT, PATCH, DELETE |
| 2 | Status codes — 2xx success, 4xx client errors, 5xx server errors |
| 3 | `requests.get()` with `params`, `headers`, and `timeout` |
| 4 | `response.raise_for_status()` and JSON parsing |
| 5 | Graceful error handling — `HTTPError`, `ConnectionError`, `Timeout` |
| 6 | Real API: **Open-Meteo** — live weather, no API key required |
| 7 | Real API: **JSONPlaceholder** — REST test endpoints (posts, users) |

---

## 🚀 How to Run

```bash
pip install requests
python "Week 1/Day 4/api_requests.py"
```

> Requires an internet connection. The script degrades gracefully if offline — all errors are caught and logged without crashing.

---

## 🧩 Script Structure

```
api_requests.py
│
├── safe_get(url, params)         # reusable GET wrapper with full error handling
├── explain_http_basics()         # HTTP verbs, status codes, request anatomy
├── fetch_weather()               # Open-Meteo: live weather for Islamabad
├── fetch_posts(limit)            # JSONPlaceholder: paginated blog posts
└── fetch_user(user_id)           # JSONPlaceholder: single user profile
```

---

## 💡 Key Concepts Demonstrated

### Reusable `safe_get` Wrapper
```python
def safe_get(url: str, params: dict | None = None) -> dict | None:
    headers = {"Accept": "application/json", "User-Agent": "AlAziz-Internship/1.0"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()   # raises HTTPError on 4xx / 5xx
        return response.json()
    except HTTPError as exc:
        logger.error("HTTP %s — %s", exc.response.status_code, exc)
    except ConnectionError:
        logger.error("No internet connection.")
    except Timeout:
        logger.error("Request timed out.")
    return None
```

### Open-Meteo — Live Weather API
```python
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 33.6844,       # Islamabad
    "longitude": 73.0479,
    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
    "timezone": "Asia/Karachi",
}
data = safe_get(url, params=params)
temp = data["current"]["temperature_2m"]   # e.g. 32.4 °C
```

### JSONPlaceholder — REST Test API
```python
# GET a list of resources
posts = safe_get("https://jsonplaceholder.typicode.com/posts")

# GET a single resource by ID
user = safe_get("https://jsonplaceholder.typicode.com/users/1")
```

### HTTP Reference Card (printed at runtime)
```
HTTP Verbs
  GET     — Retrieve a resource (read-only, cacheable)
  POST    — Create a new resource (send body)
  PUT     — Replace an existing resource
  PATCH   — Partially update a resource
  DELETE  — Remove a resource

Status Codes
  ✓ 200 — OK
  ✓ 201 — Created
  ✗ 400 — Bad Request
  ✗ 401 — Unauthorized
  ✗ 404 — Not Found
  ✗ 429 — Too Many Requests
  ✗ 500 — Internal Server Error
```

---

## 📊 Sample Output

```
── Weather: Islamabad, Pakistan (Open-Meteo) ──
  Temperature  : 32.4 °C
  Humidity     : 48 %
  Wind Speed   : 14.2 km/h
  Condition    : Partly cloudy
  Time         : 2025-08-15T14:00

── User #2 Profile ──
  Name     : Ervin Howell
  Username : Antonette
  Email    : Shanna@melissa.tv
  Company  : Deckow-Crist — Proactive didactic contingency
  City     : Wisokyburgh
```

---

## 🛠️ Skills Demonstrated

`REST APIs` · `Python requests` · `HTTP Protocol` · `JSON Parsing` · `Error Handling` · `API Integration` · `Real-time Data Fetching` · `Production Patterns`

---

## 👤 About Me

**Fatir Faraz** — Final-year Software Engineering student specializing in AI & Automation Engineering.
Currently interning at **Al Aziz Technologies** on the AI Engineering track, building hands-on experience across Python, ML pipelines, and AI agent development.

- 🔗 GitHub: [@Fatir002](https://github.com/Fatir002)
- 💼 LinkedIn: [linkedin.com/in/fatirfaraz](https://linkedin.com/in/fatirfaraz)
- 🌐 Fiverr: AI Automation & WhatsApp Agent Services
