"""
Week 1 | Day 5 — Capstone: CLI Analytics Tool
==============================================
Internship at Al Aziz Technologies | AI Engineering Track

Architecture:
    DataAnalyser     — loads inline employee dataset, computes stats
    WeatherClient    — fetches live weather via Open-Meteo (no API key)
    ReportGenerator  — combines outputs into a formatted text report
    CLI entry point  — argparse routes subcommands

Commands:
    python capstone_app.py stats            # Analyse employee dataset
    python capstone_app.py weather          # Fetch live Islamabad weather
    python capstone_app.py weather --city Tokyo --lat 35.68 --lon 139.69
    python capstone_app.py report           # Combined stats + weather report
    python capstone_app.py --help

Design principles applied this week:
    ✓ OOP (classes with properties and dunder methods)
    ✓ Type hints throughout
    ✓ Structured logging
    ✓ Robust error handling (try/except at every I/O boundary)
    ✓ Modular functions — each class is independently testable
    ✓ Self-contained inline data — no external file dependencies
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

try:
    import requests
    from requests.exceptions import ConnectionError, HTTPError, Timeout
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("capstone")

BANNER = """
╔══════════════════════════════════════════════════════════╗
║   Al Aziz Technologies — Week 1 Capstone                 ║
║   CLI Analytics Tool  v1.0                               ║
║   Intern: Fatir Faraz  |  AI Engineering Track           ║
╚══════════════════════════════════════════════════════════╝
"""


# ══════════════════════════════════════════════════════════════════════════════
# INLINE DATASET
# ══════════════════════════════════════════════════════════════════════════════
EMPLOYEES: list[dict[str, Any]] = [
    {"id": 1,  "name": "Ayesha Malik",    "dept": "AI Research",  "salary": 95_000, "exp_years": 6,  "score": 91},
    {"id": 2,  "name": "Bilal Khan",      "dept": "Data Science", "salary": 82_000, "exp_years": 4,  "score": 85},
    {"id": 3,  "name": "Chand Rehman",    "dept": "AI Research",  "salary": 91_000, "exp_years": 5,  "score": 88},
    {"id": 4,  "name": "Dania Siddiqui",  "dept": "Engineering",  "salary": 78_000, "exp_years": 3,  "score": 80},
    {"id": 5,  "name": "Ehsan Tariq",     "dept": "Data Science", "salary": 88_000, "exp_years": 5,  "score": 87},
    {"id": 6,  "name": "Fatima Noor",     "dept": "Engineering",  "salary": 72_000, "exp_years": 2,  "score": 76},
    {"id": 7,  "name": "Ghazal Butt",     "dept": "AI Research",  "salary": 105_000,"exp_years": 9,  "score": 95},
    {"id": 8,  "name": "Hamza Ali",       "dept": "Data Science", "salary": 79_000, "exp_years": 3,  "score": 81},
    {"id": 9,  "name": "Iqra Shahid",     "dept": "Engineering",  "salary": 68_000, "exp_years": 1,  "score": 73},
    {"id": 10, "name": "Jawad Mehmood",   "dept": "AI Research",  "salary": 98_000, "exp_years": 7,  "score": 93},
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Stats:
    """Container for descriptive statistics of a numeric field."""
    field_name: str
    count: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    total: float

    def __str__(self) -> str:
        return (
            f"  {self.field_name:<14} "
            f"mean={self.mean:>10,.2f}  "
            f"std={self.std:>9,.2f}  "
            f"min={self.minimum:>9,.2f}  "
            f"max={self.maximum:>9,.2f}"
        )


@dataclass
class WeatherData:
    """Parsed weather response."""
    city: str
    temperature: float
    humidity: float
    wind_speed: float
    condition: str
    fetched_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ══════════════════════════════════════════════════════════════════════════════
# CLASS: DataAnalyser
# ══════════════════════════════════════════════════════════════════════════════
class DataAnalyser:
    """Load and analyse the employee dataset.

    Args:
        records: List of employee dicts.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        if not records:
            raise ValueError("DataAnalyser requires at least one record.")
        self._records = records
        logger.info("DataAnalyser initialised with %d records.", len(records))

    # ── public interface ──
    def compute_stats(self, field: str) -> Stats:
        """Compute descriptive statistics for a numeric field."""
        try:
            values = np.array([r[field] for r in self._records], dtype=float)
        except KeyError:
            raise KeyError(f"Field {field!r} not found in records.")
        return Stats(
            field_name=field,
            count=int(len(values)),
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            std=float(np.std(values)),
            minimum=float(np.min(values)),
            maximum=float(np.max(values)),
            total=float(np.sum(values)),
        )

    def dept_breakdown(self) -> dict[str, dict[str, Any]]:
        """Return per-department headcount and average salary."""
        depts: dict[str, list[float]] = {}
        for r in self._records:
            depts.setdefault(r["dept"], []).append(r["salary"])
        return {
            dept: {
                "headcount": len(salaries),
                "avg_salary": round(sum(salaries) / len(salaries), 2),
                "total_payroll": sum(salaries),
            }
            for dept, salaries in depts.items()
        }

    def top_performers(self, n: int = 3, by: str = "score") -> list[dict[str, Any]]:
        """Return the top-n employees ranked by a given field."""
        return sorted(self._records, key=lambda r: r.get(by, 0), reverse=True)[:n]

    def correlation(self, field_a: str, field_b: str) -> float:
        """Return Pearson correlation coefficient between two numeric fields."""
        a = np.array([r[field_a] for r in self._records], dtype=float)
        b = np.array([r[field_b] for r in self._records], dtype=float)
        return float(np.corrcoef(a, b)[0, 1])

    def print_table(self) -> None:
        """Print the full dataset as a formatted table."""
        header = f"  {'ID':>3}  {'Name':<22}  {'Dept':<15}  {'Salary':>10}  {'Exp':>4}  {'Score':>5}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in self._records:
            print(
                f"  {r['id']:>3}  {r['name']:<22}  {r['dept']:<15}  "
                f"PKR {r['salary']:>7,}  {r['exp_years']:>3}y  {r['score']:>5}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# CLASS: WeatherClient
# ══════════════════════════════════════════════════════════════════════════════
class WeatherClient:
    """Fetch current weather from Open-Meteo (free, no API key).

    Args:
        city: Display name for the location.
        lat: Latitude.
        lon: Longitude.
        timeout: Request timeout in seconds.
    """

    _BASE_URL = "https://api.open-meteo.com/v1/forecast"
    _WMO_CODES: dict[int, str] = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Icy fog", 51: "Light drizzle", 53: "Moderate drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 80: "Rain showers", 95: "Thunderstorm",
    }

    def __init__(self, city: str = "Islamabad", lat: float = 33.6844, lon: float = 73.0479,
                 timeout: int = 10) -> None:
        self.city = city
        self.lat = lat
        self.lon = lon
        self.timeout = timeout

    def fetch(self) -> WeatherData | None:
        """Fetch current weather and return a WeatherData object."""
        if not _REQUESTS_OK:
            logger.error("'requests' not installed — cannot fetch weather.")
            return None

        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        }
        logger.info("Fetching weather for %s (%.4f, %.4f)…", self.city, self.lat, self.lon)
        try:
            t0 = time.perf_counter()
            resp = requests.get(self._BASE_URL, params=params, timeout=self.timeout)
            elapsed = time.perf_counter() - t0
            resp.raise_for_status()
            logger.info("Weather response: HTTP 200 in %.2fs", elapsed)
            cur = resp.json().get("current", {})
            code = cur.get("weather_code", -1)
            return WeatherData(
                city=self.city,
                temperature=cur.get("temperature_2m", float("nan")),
                humidity=cur.get("relative_humidity_2m", float("nan")),
                wind_speed=cur.get("wind_speed_10m", float("nan")),
                condition=self._WMO_CODES.get(code, f"WMO {code}"),
            )
        except HTTPError as exc:
            logger.error("HTTP error: %s", exc)
        except ConnectionError:
            logger.error("No internet connection.")
        except Timeout:
            logger.error("Request timed out after %ds.", self.timeout)
        except Exception as exc:
            logger.error("Unexpected error: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CLASS: ReportGenerator
# ══════════════════════════════════════════════════════════════════════════════
class ReportGenerator:
    """Combine DataAnalyser and WeatherClient outputs into a formatted report."""

    def __init__(self, analyser: DataAnalyser, weather_client: WeatherClient) -> None:
        self._analyser = analyser
        self._weather_client = weather_client

    def generate(self) -> str:
        """Build and return the full report as a string."""
        lines: list[str] = []
        sep = "=" * 62

        lines.append(sep)
        lines.append("  AL AZIZ TECHNOLOGIES — INTERNSHIP ANALYTICS REPORT")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(sep)

        # Section 1: Employee stats
        lines.append("\n  SECTION 1 — EMPLOYEE STATISTICS\n")
        for fld in ["salary", "exp_years", "score"]:
            s = self._analyser.compute_stats(fld)
            lines.append(str(s))

        lines.append(f"\n  Salary / Experience correlation : "
                     f"{self._analyser.correlation('salary', 'exp_years'):.4f}")
        lines.append(f"  Salary / Score correlation      : "
                     f"{self._analyser.correlation('salary', 'score'):.4f}")

        # Section 2: Dept breakdown
        lines.append("\n  SECTION 2 — DEPARTMENT BREAKDOWN\n")
        breakdown = self._analyser.dept_breakdown()
        for dept, info in sorted(breakdown.items()):
            lines.append(
                f"  {dept:<16}  headcount={info['headcount']}  "
                f"avg_salary=PKR {info['avg_salary']:>9,.0f}  "
                f"payroll=PKR {info['total_payroll']:>10,}"
            )

        # Section 3: Top performers
        lines.append("\n  SECTION 3 — TOP 3 PERFORMERS (by score)\n")
        for rank, emp in enumerate(self._analyser.top_performers(3, by="score"), 1):
            lines.append(
                f"  #{rank} {emp['name']:<22} score={emp['score']}  "
                f"dept={emp['dept']}  salary=PKR {emp['salary']:,}"
            )

        # Section 4: Weather
        lines.append("\n  SECTION 4 — LIVE WEATHER\n")
        weather = self._weather_client.fetch()
        if weather:
            lines.append(f"  City        : {weather.city}")
            lines.append(f"  Temperature : {weather.temperature} °C")
            lines.append(f"  Humidity    : {weather.humidity} %")
            lines.append(f"  Wind Speed  : {weather.wind_speed} km/h")
            lines.append(f"  Condition   : {weather.condition}")
            lines.append(f"  Fetched At  : {weather.fetched_at}")
        else:
            lines.append("  Weather data unavailable — check internet connection.")

        lines.append("\n" + sep)
        lines.append("  END OF REPORT")
        lines.append(sep)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CLI COMMANDS
# ══════════════════════════════════════════════════════════════════════════════
def cmd_stats(_args: argparse.Namespace) -> int:
    """Run the stats command."""
    print(BANNER)
    print("── Employee Dataset ──")
    analyser = DataAnalyser(EMPLOYEES)
    analyser.print_table()

    print("\n── Descriptive Statistics ──")
    for fld in ["salary", "exp_years", "score"]:
        s = analyser.compute_stats(fld)
        print(s)

    print("\n── Department Breakdown ──")
    breakdown = analyser.dept_breakdown()
    for dept, info in sorted(breakdown.items()):
        print(
            f"  {dept:<16}  headcount={info['headcount']}  "
            f"avg_salary=PKR {info['avg_salary']:>9,.0f}"
        )

    print("\n── Top 3 Performers (by score) ──")
    for rank, emp in enumerate(analyser.top_performers(3), 1):
        print(f"  #{rank} {emp['name']:<22} score={emp['score']}")

    print("\n── Correlations ──")
    print(f"  Salary ↔ Experience : {analyser.correlation('salary', 'exp_years'):.4f}")
    print(f"  Salary ↔ Score      : {analyser.correlation('salary', 'score'):.4f}")
    return 0


def cmd_weather(args: argparse.Namespace) -> int:
    """Run the weather command."""
    print(BANNER)
    client = WeatherClient(city=args.city, lat=args.lat, lon=args.lon)
    weather = client.fetch()
    if weather is None:
        print("  ✗ Could not retrieve weather data.")
        return 1
    print(f"── Weather: {weather.city} ──")
    print(f"  Temperature : {weather.temperature} °C")
    print(f"  Humidity    : {weather.humidity} %")
    print(f"  Wind        : {weather.wind_speed} km/h")
    print(f"  Condition   : {weather.condition}")
    print(f"  Fetched At  : {weather.fetched_at}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Run the full report command."""
    print(BANNER)
    analyser = DataAnalyser(EMPLOYEES)
    weather_client = WeatherClient(city=args.city, lat=args.lat, lon=args.lon)
    reporter = ReportGenerator(analyser, weather_client)
    report = reporter.generate()
    print(report)
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capstone_app",
        description="Al Aziz Technologies — Week 1 Capstone CLI Analytics Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python capstone_app.py stats
  python capstone_app.py weather
  python capstone_app.py weather --city Tokyo --lat 35.68 --lon 139.69
  python capstone_app.py report
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # stats
    subparsers.add_parser("stats", help="Analyse the inline employee dataset.")

    # weather (shared args)
    weather_args = argparse.ArgumentParser(add_help=False)
    weather_args.add_argument("--city", default="Islamabad", help="City display name")
    weather_args.add_argument("--lat",  default=33.6844, type=float, help="Latitude")
    weather_args.add_argument("--lon",  default=73.0479, type=float, help="Longitude")

    subparsers.add_parser("weather", parents=[weather_args],
                          help="Fetch live weather from Open-Meteo.")
    subparsers.add_parser("report",  parents=[weather_args],
                          help="Generate combined analytics + weather report.")
    return parser


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "stats":   cmd_stats,
        "weather": cmd_weather,
        "report":  cmd_report,
    }

    try:
        exit_code = dispatch[args.command](args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.critical("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
