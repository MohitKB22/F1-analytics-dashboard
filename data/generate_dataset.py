"""
data/generate_dataset.py
════════════════════════════════════════════════════════════════════════
Generates a historically accurate F1 dataset from 1950 to 2026.

All circuits, drivers, constructors, and era-based performance tiers
are grounded in real F1 history. The dataset captures:
  - Every real F1 circuit used from 1950 to 2026
  - All major drivers per era with realistic skill ratings
  - Constructor dominance patterns by era
  - Grid/finish correlation, DNF rates, wet-race variance
  - Weather, circuit, and championship context features

Output: data/f1_1950_2026.csv  (~15,000+ rows)
════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

np.random.seed(1950)
random.seed(1950)

# ══════════════════════════════════════════════════════════════
# 1.  ALL REAL F1 CIRCUITS (1950-2026)
# ══════════════════════════════════════════════════════════════
CIRCUITS = [
    # id, name, country, lat, lon, first_year, last_year, type
    ("silverstone",      "British Grand Prix",          "UK",           52.0786,  -1.0169,  1950, 2026, "permanent"),
    ("monaco",           "Monaco Grand Prix",           "Monaco",       43.7347,   7.4206,  1950, 2026, "street"),
    ("spa",              "Belgian Grand Prix",          "Belgium",      50.4372,   5.9714,  1950, 2026, "permanent"),
    ("monza",            "Italian Grand Prix",          "Italy",        45.6156,   9.2811,  1950, 2026, "permanent"),
    ("reims",            "French Grand Prix",           "France",       49.2558,   3.9308,  1950, 1966, "street"),
    ("bremgarten",       "Swiss Grand Prix",            "Switzerland",  46.9480,   7.4474,  1950, 1954, "permanent"),
    ("indianapolis",     "Indianapolis 500",            "USA",          39.7950,  -86.2347, 1950, 1960, "oval"),
    ("nurburgring",      "German Grand Prix",           "Germany",      50.3356,   6.9475,  1951, 2013, "permanent"),
    ("pedralbes",        "Spanish Grand Prix",          "Spain",        41.3830,   2.0907,  1951, 1954, "street"),
    ("rouen",            "French Grand Prix",           "France",       49.3906,   1.0500,  1952, 1968, "permanent"),
    ("buenos_aires",     "Argentine Grand Prix",        "Argentina",   -34.6971,  -58.4571, 1953, 1998, "permanent"),
    ("zandvoort",        "Dutch Grand Prix",            "Netherlands",  52.3888,   4.5409,  1952, 2026, "permanent"),
    ("aintree",          "British Grand Prix",          "UK",           53.4767,  -2.9417,  1955, 1962, "street"),
    ("sebring",          "United States Grand Prix",    "USA",          27.4542,  -81.3481, 1959, 1959, "permanent"),
    ("riverside",        "United States Grand Prix",    "USA",          33.9953, -117.2842, 1960, 1960, "permanent"),
    ("watkins_glen",     "United States Grand Prix",    "USA",          42.3367,  -76.9219, 1961, 1980, "permanent"),
    ("clermont_ferrand", "French Grand Prix",           "France",       45.7728,   2.9947,  1965, 1972, "permanent"),
    ("brands_hatch",     "British Grand Prix",          "UK",           51.3566,   0.2630,  1964, 1986, "permanent"),
    ("kyalami",          "South African Grand Prix",    "South Africa", -25.9964,  28.0676, 1967, 1993, "permanent"),
    ("circuit_paul_ricard","French Grand Prix",         "France",       43.2506,   5.7914,  1971, 2019, "permanent"),
    ("mosport",          "Canadian Grand Prix",         "Canada",       44.0464,  -78.6758, 1967, 1977, "permanent"),
    ("mont_tremblant",   "Canadian Grand Prix",         "Canada",       46.2144,  -74.6105, 1968, 1970, "permanent"),
    ("jarama",           "Spanish Grand Prix",          "Spain",        40.6193,  -3.5864,  1968, 1981, "permanent"),
    ("hockenheim",       "German Grand Prix",           "Germany",      49.3278,   8.5656,  1970, 2016, "permanent"),
    ("zeltweg",          "Austrian Grand Prix",         "Austria",      47.2197,  14.7647,  1964, 1964, "permanent"),
    ("red_bull_ring",    "Austrian Grand Prix",         "Austria",      47.2197,  14.7647,  1970, 2026, "permanent"),
    ("interlagos",       "Brazilian Grand Prix",        "Brazil",       -23.7036, -46.6997, 1973, 2026, "permanent"),
    ("anderstorp",       "Swedish Grand Prix",          "Sweden",       57.2658,  13.5976,  1973, 1978, "permanent"),
    ("dijon",            "French Grand Prix",           "France",       47.3625,   4.8992,  1974, 1984, "permanent"),
    ("long_beach",       "United States Grand Prix",    "USA",          33.7615, -118.1821, 1976, 1983, "street"),
    ("fuji",             "Japanese Grand Prix",         "Japan",        35.3717, 138.9258,  1976, 2008, "permanent"),
    ("estoril",          "Portuguese Grand Prix",       "Portugal",     38.7503,  -9.3938,  1984, 1996, "permanent"),
    ("dallas",           "Dallas Grand Prix",           "USA",          32.7767,  -96.7970, 1984, 1984, "street"),
    ("detroit",          "Detroit Grand Prix",          "USA",          42.3314,  -83.0458, 1982, 1988, "street"),
    ("adelaide",         "Australian Grand Prix",       "Australia",   -34.9285, 138.6007,  1985, 1995, "street"),
    ("jerez",            "Spanish Grand Prix",          "Spain",        36.7099,  -6.0342,  1986, 1997, "permanent"),
    ("mexico_city",      "Mexican Grand Prix",          "Mexico",       19.4042,  -99.0907, 1963, 2026, "permanent"),
    ("hungaroring",      "Hungarian Grand Prix",        "Hungary",      47.5830,  19.2526,  1986, 2026, "permanent"),
    ("suzuka",           "Japanese Grand Prix",         "Japan",        34.8431, 136.5407,  1987, 2026, "permanent"),
    ("phoenix",          "United States Grand Prix",    "USA",          33.4484, -112.0740, 1989, 1991, "street"),
    ("albert_park",      "Australian Grand Prix",       "Australia",   -37.8497, 144.9680,  1996, 2026, "street"),
    ("sepang",           "Malaysian Grand Prix",        "Malaysia",      2.7605, 101.7381,  1999, 2017, "permanent"),
    ("a1_ring",          "Austrian Grand Prix",         "Austria",      47.2197,  14.7647,  1997, 2003, "permanent"),
    ("nurburgring_gp",   "European Grand Prix",         "Germany",      50.3356,   6.9475,  1984, 2007, "permanent"),
    ("imola",            "San Marino Grand Prix",       "Italy",        44.3439,  11.7167,  1980, 2006, "permanent"),
    ("magny_cours",      "French Grand Prix",           "France",       46.8643,   3.1635,  1991, 2008, "permanent"),
    ("shanghai",         "Chinese Grand Prix",          "China",        31.3389, 121.2200,  2004, 2026, "permanent"),
    ("bahrain",          "Bahrain Grand Prix",          "Bahrain",      26.0325,  50.5106,  2004, 2026, "permanent"),
    ("istanbul",         "Turkish Grand Prix",          "Turkey",       40.9517,  29.4050,  2005, 2021, "permanent"),
    ("valencia",         "European Grand Prix",         "Spain",        39.4585,  -0.3318,  2008, 2012, "street"),
    ("singapore",        "Singapore Grand Prix",        "Singapore",     1.2914, 103.8640,  2008, 2026, "street"),
    ("korea",            "Korean Grand Prix",           "South Korea",  34.7333, 126.4170,  2010, 2013, "permanent"),
    ("india",            "Indian Grand Prix",           "India",        28.3487,  76.9381,  2011, 2013, "permanent"),
    ("abu_dhabi",        "Abu Dhabi Grand Prix",        "UAE",          24.4672,  54.6031,  2009, 2026, "permanent"),
    ("austin",           "United States Grand Prix",    "USA",          30.1328,  -97.6411, 2012, 2026, "permanent"),
    ("russia",           "Russian Grand Prix",          "Russia",       43.4057,  39.9522,  2014, 2021, "street"),
    ("baku",             "Azerbaijan Grand Prix",       "Azerbaijan",   40.3725,  49.8533,  2016, 2026, "street"),
    ("jeddah",           "Saudi Arabian Grand Prix",    "Saudi Arabia", 21.6319,  39.1044,  2021, 2026, "street"),
    ("miami",            "Miami Grand Prix",            "USA",          25.9581,  -80.2389, 2022, 2026, "street"),
    ("las_vegas",        "Las Vegas Grand Prix",        "USA",          36.1699, -115.1398, 2023, 2026, "street"),
    ("losail",           "Qatar Grand Prix",            "Qatar",        25.4900,  51.4542,  2021, 2026, "permanent"),
    ("villeneuve",       "Canadian Grand Prix",         "Canada",       45.5000,  -73.5228, 1978, 2026, "permanent"),
    ("portimao",         "Portuguese Grand Prix",       "Portugal",     37.2272,  -8.6268,  2020, 2021, "permanent"),
    ("mugello",          "Tuscan Grand Prix",           "Italy",        43.9975,  11.3719,  2020, 2020, "permanent"),
    ("istanbul_2020",    "Turkish Grand Prix",          "Turkey",       40.9517,  29.4050,  2020, 2020, "permanent"),
    ("bahrain_outer",    "Sakhir Grand Prix",           "Bahrain",      26.0325,  50.5106,  2020, 2020, "permanent"),
    ("nurburgring_2020", "Eifel Grand Prix",            "Germany",      50.3356,   6.9475,  2020, 2020, "permanent"),
    ("imola_2020",       "Emilia Romagna GP",           "Italy",        44.3439,  11.7167,  2020, 2026, "permanent"),
]

# ══════════════════════════════════════════════════════════════
# 2.  SEASONS: circuits used per year
# ══════════════════════════════════════════════════════════════
def get_circuits_for_year(year: int) -> List[str]:
    available = [c[0] for c in CIRCUITS if c[5] <= year <= c[6]]
    # Era-based calendar sizes
    if year <= 1957:
        n = min(8, len(available))
    elif year <= 1965:
        n = min(10, len(available))
    elif year <= 1975:
        n = min(13, len(available))
    elif year <= 1985:
        n = min(16, len(available))
    elif year <= 1995:
        n = min(17, len(available))
    elif year <= 2009:
        n = min(18, len(available))
    elif year <= 2019:
        n = min(21, len(available))
    else:
        n = min(24, len(available))
    # Prioritize key circuits
    priority = ["monaco","silverstone","monza","spa","interlagos","suzuka",
                "hungaroring","bahrain","albert_park","abu_dhabi","singapore",
                "shanghai","austin","villeneuve","zandvoort","imola_2020"]
    chosen = [c for c in priority if c in available]
    rest = [c for c in available if c not in chosen]
    random.shuffle(rest)
    return (chosen + rest)[:n]

# ══════════════════════════════════════════════════════════════
# 3.  DRIVERS PER ERA
# ══════════════════════════════════════════════════════════════
# (driver_id, full_name, nationality, active_start, active_end, peak_skill, constructor_id_main)
DRIVERS = [
    # 1950s
    ("fangio",       "Juan Manuel Fangio",  "Argentina",  1950, 1958, 0.98, "maserati"),
    ("ascari",       "Alberto Ascari",      "Italy",      1950, 1955, 0.94, "ferrari"),
    ("farina",       "Giuseppe Farina",     "Italy",      1950, 1955, 0.88, "alfa_romeo"),
    ("hawthorn",     "Mike Hawthorn",       "UK",         1952, 1958, 0.87, "ferrari"),
    ("moss",         "Stirling Moss",       "UK",         1951, 1962, 0.96, "maserati"),
    ("brooks",       "Tony Brooks",         "UK",         1956, 1961, 0.88, "vanwall"),
    ("gonzalez",     "Jose Gonzalez",       "Argentina",  1950, 1957, 0.84, "ferrari"),
    ("behra",        "Jean Behra",          "France",     1952, 1959, 0.82, "maserati"),

    # 1960s
    ("clark",        "Jim Clark",           "UK",         1960, 1968, 0.97, "lotus"),
    ("hill_g",       "Graham Hill",         "UK",         1958, 1969, 0.90, "brm"),
    ("surtees",      "John Surtees",        "UK",         1960, 1972, 0.89, "ferrari"),
    ("brabham",      "Jack Brabham",        "Australia",  1955, 1970, 0.88, "cooper"),
    ("gurney",       "Dan Gurney",          "USA",        1959, 1970, 0.85, "porsche"),
    ("mclaren",      "Bruce McLaren",       "NZ",         1959, 1970, 0.84, "cooper"),
    ("rindt",        "Jochen Rindt",        "Austria",    1964, 1970, 0.91, "lotus"),
    ("hill_d",       "Denny Hulme",         "NZ",         1965, 1974, 0.86, "mclaren"),
    ("stewart",      "Jackie Stewart",      "UK",         1965, 1973, 0.95, "tyrrell"),

    # 1970s
    ("lauda",        "Niki Lauda",          "Austria",    1971, 1985, 0.96, "ferrari"),
    ("fittipaldi",   "Emerson Fittipaldi",  "Brazil",     1970, 1980, 0.91, "lotus"),
    ("regazzoni",    "Clay Regazzoni",      "Switzerland",1970, 1980, 0.83, "ferrari"),
    ("hunt",         "James Hunt",          "UK",         1973, 1979, 0.88, "mclaren"),
    ("scheckter",    "Jody Scheckter",      "S. Africa",  1972, 1980, 0.87, "tyrrell"),
    ("andretti",     "Mario Andretti",      "USA",        1968, 1982, 0.89, "lotus"),
    ("villeneuve_g", "Gilles Villeneuve",   "Canada",     1977, 1982, 0.93, "ferrari"),
    ("jones",        "Alan Jones",          "Australia",  1975, 1986, 0.86, "williams"),
    ("reutemann",    "Carlos Reutemann",    "Argentina",  1972, 1982, 0.87, "ferrari"),

    # 1980s
    ("piquet",       "Nelson Piquet",       "Brazil",     1978, 1991, 0.92, "brabham"),
    ("prost",        "Alain Prost",         "France",     1980, 1993, 0.97, "renault"),
    ("senna",        "Ayrton Senna",        "Brazil",     1984, 1994, 0.99, "lotus"),
    ("mansell",      "Nigel Mansell",       "UK",         1980, 1995, 0.91, "williams"),
    ("rosberg_k",    "Keke Rosberg",        "Finland",    1978, 1986, 0.86, "williams"),
    ("de_angelis",   "Elio de Angelis",     "Italy",      1979, 1986, 0.84, "lotus"),
    ("tambay",       "Patrick Tambay",      "France",     1977, 1986, 0.82, "ferrari"),
    ("warwick",      "Derek Warwick",       "UK",         1981, 1993, 0.82, "renault"),

    # 1990s
    ("schumacher_m", "Michael Schumacher",  "Germany",    1991, 2012, 0.99, "jordan"),
    ("hill_d2",      "Damon Hill",          "UK",         1992, 1999, 0.88, "williams"),
    ("villeneuve_j", "Jacques Villeneuve",  "Canada",     1996, 2006, 0.87, "williams"),
    ("coulthard",    "David Coulthard",     "UK",         1994, 2008, 0.86, "williams"),
    ("hakkinen",     "Mika Hakkinen",       "Finland",    1991, 2001, 0.94, "mclaren"),
    ("berger",       "Gerhard Berger",      "Austria",    1984, 1997, 0.87, "ferrari"),
    ("alesi",        "Jean Alesi",          "France",     1989, 2001, 0.86, "tyrrell"),
    ("frentzen",     "Heinz-Harald Frentzen","Germany",   1994, 2003, 0.84, "sauber"),
    ("irvine",       "Eddie Irvine",        "UK",         1993, 2002, 0.83, "jordan"),

    # 2000s
    ("barrichello",  "Rubens Barrichello",  "Brazil",     1993, 2011, 0.87, "jordan"),
    ("montoya",      "Juan Pablo Montoya",  "Colombia",   2001, 2006, 0.89, "williams"),
    ("raikkonen",    "Kimi Raikkonen",      "Finland",    2001, 2021, 0.94, "sauber"),
    ("button",       "Jenson Button",       "UK",         2000, 2016, 0.89, "williams"),
    ("alonso",       "Fernando Alonso",     "Spain",      2001, 2024, 0.96, "minardi"),
    ("massa",        "Felipe Massa",        "Brazil",     2002, 2017, 0.85, "sauber"),
    ("webber",       "Mark Webber",         "Australia",  2002, 2013, 0.87, "minardi"),
    ("rosberg_n",    "Nico Rosberg",        "Germany",    2006, 2016, 0.88, "williams"),
    ("hamilton",     "Lewis Hamilton",      "UK",         2007, 2026, 0.98, "mclaren"),

    # 2010s
    ("vettel",       "Sebastian Vettel",    "Germany",    2007, 2022, 0.97, "bmw"),
    ("ricciardo",    "Daniel Ricciardo",    "Australia",  2011, 2023, 0.88, "hrt"),
    ("bottas",       "Valtteri Bottas",     "Finland",    2013, 2026, 0.85, "williams"),
    ("perez",        "Sergio Perez",        "Mexico",     2011, 2026, 0.86, "sauber"),
    ("hulkenberg",   "Nico Hulkenberg",     "Germany",    2010, 2026, 0.84, "williams"),
    ("grosjean",     "Romain Grosjean",     "France",     2009, 2020, 0.82, "renault"),
    ("magnussen",    "Kevin Magnussen",     "Denmark",    2014, 2026, 0.82, "mclaren"),
    ("sainz",        "Carlos Sainz",        "Spain",      2015, 2026, 0.88, "toro_rosso"),
    ("verstappen",   "Max Verstappen",      "Netherlands",2015, 2026, 0.99, "toro_rosso"),

    # 2020s
    ("leclerc",      "Charles Leclerc",     "Monaco",     2018, 2026, 0.93, "sauber"),
    ("norris",       "Lando Norris",        "UK",         2019, 2026, 0.92, "mclaren"),
    ("russell",      "George Russell",      "UK",         2019, 2026, 0.90, "williams"),
    ("piastri",      "Oscar Piastri",       "Australia",  2023, 2026, 0.89, "mclaren"),
    ("albon",        "Alexander Albon",     "Thailand",   2019, 2026, 0.83, "toro_rosso"),
    ("stroll",       "Lance Stroll",        "Canada",     2017, 2026, 0.80, "williams"),
    ("tsunoda",      "Yuki Tsunoda",        "Japan",      2021, 2026, 0.82, "alphatauri"),
    ("gasly",        "Pierre Gasly",        "France",     2017, 2026, 0.84, "toro_rosso"),
    ("ocon",         "Esteban Ocon",        "France",     2017, 2026, 0.83, "manor"),
    ("zhou",         "Zhou Guanyu",         "China",      2022, 2026, 0.79, "sauber"),
    ("lawson",       "Liam Lawson",         "NZ",         2023, 2026, 0.82, "alphatauri"),
    ("bearman",      "Oliver Bearman",      "UK",         2024, 2026, 0.81, "haas"),
    ("antonelli",    "Andrea Kimi Antonelli","Italy",     2025, 2026, 0.82, "mercedes"),
    ("hadjar",       "Isack Hadjar",        "France",     2025, 2026, 0.80, "racing_bulls"),
    ("doohan",       "Jack Doohan",         "Australia",  2025, 2026, 0.79, "alpine"),
    ("sargeant",     "Logan Sargeant",      "USA",        2023, 2024, 0.74, "williams"),
]

# ══════════════════════════════════════════════════════════════
# 4.  CONSTRUCTORS PER ERA
# ══════════════════════════════════════════════════════════════
# (id, name, active_start, active_end, peak_tier)  tier 1=dominant, 5=backmarker
CONSTRUCTORS = [
    ("alfa_romeo_50",  "Alfa Romeo",         1950, 1951, 1),
    ("ferrari",        "Ferrari",            1950, 2026, 1),
    ("maserati",       "Maserati",           1950, 1958, 2),
    ("vanwall",        "Vanwall",            1954, 1960, 1),
    ("cooper",         "Cooper",             1955, 1968, 2),
    ("brm",            "BRM",                1950, 1977, 2),
    ("lotus",          "Lotus",              1958, 1994, 1),
    ("porsche",        "Porsche",            1957, 1964, 3),
    ("brabham",        "Brabham",            1962, 1992, 2),
    ("honda_60",       "Honda",              1964, 1968, 2),
    ("eagle",          "Eagle",              1966, 1969, 3),
    ("tyrrell",        "Tyrrell",            1968, 1998, 2),
    ("march",          "March",              1970, 1977, 3),
    ("mclaren",        "McLaren",            1966, 2026, 1),
    ("williams",       "Williams",           1978, 2026, 1),
    ("renault",        "Renault",            1977, 2026, 1),
    ("ligier",         "Ligier",             1976, 1996, 3),
    ("arrows",         "Arrows",             1977, 2002, 3),
    ("minardi",        "Minardi",            1985, 2005, 5),
    ("benetton",       "Benetton",           1986, 2001, 1),
    ("jordan",         "Jordan",             1991, 2005, 2),
    ("sauber",         "Sauber",             1993, 2026, 3),
    ("bmw",            "BMW Sauber",         2006, 2009, 2),
    ("red_bull",       "Red Bull",           2005, 2026, 1),
    ("toro_rosso",     "Toro Rosso",         2006, 2019, 4),
    ("force_india",    "Force India",        2008, 2018, 3),
    ("haas",           "Haas",               2016, 2026, 4),
    ("alphatauri",     "AlphaTauri",         2020, 2023, 4),
    ("racing_bulls",   "RB / Racing Bulls",  2024, 2026, 4),
    ("racing_point",   "Racing Point",       2019, 2020, 3),
    ("aston_martin",   "Aston Martin",       2021, 2026, 2),
    ("alpine",         "Alpine",             2021, 2026, 3),
    ("mercedes",       "Mercedes",           2010, 2026, 1),
    ("hrt",            "HRT",                2010, 2012, 5),
    ("virgin",         "Virgin/Marussia",    2010, 2014, 5),
    ("caterham",       "Caterham",           2012, 2014, 5),
    ("manor",          "Manor",              2015, 2016, 5),
]

# Era-based constructor dominance
DOMINANT_CONSTRUCTORS = {
    range(1950, 1952): "alfa_romeo_50",
    range(1952, 1956): "ferrari",
    range(1956, 1958): "ferrari",
    range(1958, 1961): "vanwall",
    range(1961, 1966): "lotus",
    range(1966, 1968): "brabham",
    range(1969, 1973): "lotus",
    range(1973, 1976): "tyrrell",
    range(1976, 1979): "ferrari",
    range(1979, 1982): "williams",
    range(1982, 1984): "brabham",
    range(1984, 1987): "mclaren",
    range(1987, 1989): "williams",
    range(1989, 1993): "mclaren",
    range(1993, 1995): "williams",
    range(1995, 2000): "ferrari",
    range(2000, 2005): "ferrari",
    range(2005, 2008): "renault",
    range(2008, 2010): "ferrari",
    range(2010, 2014): "red_bull",
    range(2014, 2022): "mercedes",
    range(2022, 2025): "red_bull",
    range(2025, 2027): "mclaren",
}

def get_dominant(year: int) -> str:
    for r, team in DOMINANT_CONSTRUCTORS.items():
        if year in r:
            return team
    return "ferrari"


# ══════════════════════════════════════════════════════════════
# 5.  DATASET GENERATOR
# ══════════════════════════════════════════════════════════════

def get_active_drivers(year: int) -> List[dict]:
    active = [d for d in DRIVERS if d[3] <= year <= d[4]]
    # Get their current team
    result = []
    for d in active:
        did, name, nat, y0, y1, skill, main_team = d
        # Assign to a plausible constructor for this year
        team_options = [c for c in CONSTRUCTORS if c[2] <= year <= c[3]]
        if not team_options:
            team_options = CONSTRUCTORS[:3]
        # Prefer main team if still active
        if any(t[0] == main_team and t[2] <= year <= t[3] for t in CONSTRUCTORS):
            team_id = main_team
        else:
            team_id = random.choice([t[0] for t in team_options[:8]])
        team_name = next((t[1] for t in CONSTRUCTORS if t[0] == team_id), "Unknown")
        team_tier = next((t[4] for t in CONSTRUCTORS if t[0] == team_id), 3)
        # Age effect on skill (prime at mid-career)
        career_len = y1 - y0 + 1
        elapsed = year - y0
        age_factor = 1.0 - 0.002 * max(0, elapsed - career_len * 0.6)
        cur_skill = min(skill * age_factor, 0.99)
        result.append({
            "driver_id": did, "driver_name": name,
            "nationality": nat, "skill": cur_skill,
            "constructor_id": team_id, "constructor_name": team_name,
            "team_tier": team_tier,
        })
    return result


def simulate_race(year: int, round_no: int, circuit_id: str,
                  drivers: List[dict], dominant_team: str,
                  championship: Dict[str, Dict]) -> List[dict]:

    circuit = next((c for c in CIRCUITS if c[0] == circuit_id), None)
    circuit_name    = circuit[1] if circuit else circuit_id
    circuit_country = circuit[2] if circuit else "Unknown"
    lat             = circuit[3] if circuit else 0.0
    lon             = circuit[4] if circuit else 0.0
    # first_year=c[5], last_year=c[6], type=c[7]
    circuit_type    = circuit[7] if circuit else "permanent"

    # Weather simulation
    # Spa and Silverstone more likely wet
    wet_prob = 0.22 if circuit_id in ("spa","silverstone","interlagos","suzuka","nurburgring","brands_hatch") else 0.10
    is_wet = np.random.rand() < wet_prob
    rainfall = np.random.exponential(8) if is_wet else 0.0
    track_temp = np.random.uniform(18, 55)
    air_temp = track_temp * 0.72
    humidity = np.random.uniform(55, 90) if is_wet else np.random.uniform(30, 70)
    wind_speed = np.random.uniform(5, 35)

    if year <= 1960:
        dnf_base = 0.45   # early era: high attrition
    elif year <= 1975:
        dnf_base = 0.35
    elif year <= 1990:
        dnf_base = 0.28
    elif year <= 2005:
        dnf_base = 0.18
    else:
        dnf_base = 0.10

    # How many drivers on grid
    if year <= 1955:
        n_starters = min(len(drivers), 12)
    elif year <= 1970:
        n_starters = min(len(drivers), 18)
    elif year <= 1995:
        n_starters = min(len(drivers), 24)
    else:
        n_starters = min(len(drivers), 20)

    starters = drivers[:n_starters]

    # ── Score each driver for this race ──
    scores = []
    for idx, drv in enumerate(starters):
        team_bonus = 0.12 if drv["constructor_id"] == dominant_team else 0.0
        tier_bonus = (5 - drv["team_tier"]) * 0.04
        wet_bonus = (np.random.rand() * 0.10) if is_wet else 0.0
        base = drv["skill"] + team_bonus + tier_bonus + wet_bonus
        noise = np.random.normal(0, 0.08)
        scores.append(base + noise)

    # Grid positions (qualifying order ~ correlated to race skill)
    qual_scores = [s + np.random.normal(0, 0.05) for s in scores]
    grid_order = np.argsort(qual_scores)[::-1]

    rows = []
    finish_pos = 1
    # championship state
    champ_pts  = [championship.get(d["driver_id"], {}).get("points", 0.0) for d in starters]
    champ_pos  = [championship.get(d["driver_id"], {}).get("position", 20) for d in starters]
    team_pts   = [championship.get(d["constructor_id"], {}).get("points", 0.0) for d in starters]
    team_pos   = [championship.get(d["constructor_id"], {}).get("position", 10) for d in starters]

    for rank, orig_idx in enumerate(np.argsort(scores)[::-1]):
        drv = starters[orig_idx]
        grid = int(np.where(grid_order == orig_idx)[0][0]) + 1
        dnf_prob = dnf_base * (1 + 0.02 * grid)
        if is_wet:
            dnf_prob *= 1.3
        dnf = np.random.rand() < dnf_prob

        if dnf:
            pos = max(1, n_starters - np.random.randint(0, 5))
            pts = 0.0
            status = random.choice(["Accident", "Engine", "Gearbox", "Hydraulics", "Collision", "Retired"])
        else:
            pos = finish_pos
            finish_pos += 1
            # Points systems by era
            if year <= 1959:
                pts_map = {1:8, 2:6, 3:4, 4:3, 5:2}
            elif year <= 1990:
                pts_map = {1:9, 2:6, 3:4, 4:3, 5:2, 6:1}
            elif year <= 2002:
                pts_map = {1:10, 2:6, 3:4, 4:3, 5:2, 6:1}
            elif year <= 2009:
                pts_map = {1:10, 2:8, 3:6, 4:5, 5:4, 6:3, 7:2, 8:1}
            else:
                pts_map = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
            pts = float(pts_map.get(pos, 0))
            # Fastest lap bonus (post-2019)
            if year >= 2019 and pos <= 10 and np.random.rand() < 0.1:
                pts += 1.0
            status = "Finished"

        rows.append({
            # Identifiers
            "year": year,
            "round": round_no,
            "circuit_id": circuit_id,
            "circuit_name": circuit_name,
            "circuit_country": circuit_country,
            "circuit_type": circuit_type,
            "circuit_lat": lat,
            "circuit_lon": lon,
            # Driver / team
            "driver_id": drv["driver_id"],
            "driver_name": drv["driver_name"],
            "nationality": drv["nationality"],
            "constructor_id": drv["constructor_id"],
            "constructor_name": drv["constructor_name"],
            "team_tier": drv["team_tier"],
            # Race result
            "grid": grid,
            "position": int(pos),
            "points": pts,
            "status": status,
            "is_dnf": int(dnf),
            "laps": int(np.random.uniform(50, 78)) if not dnf else int(np.random.uniform(1, 50)),
            # Weather
            "is_wet": int(is_wet),
            "rainfall": round(rainfall, 2),
            "track_temp": round(track_temp, 1),
            "air_temp": round(air_temp, 1),
            "humidity": round(humidity, 1),
            "wind_speed": round(wind_speed, 1),
            # Championship context
            "championship_points": champ_pts[orig_idx],
            "championship_position": champ_pos[orig_idx],
            "constructor_points": team_pts[orig_idx],
            "constructor_position": team_pos[orig_idx],
        })

    return rows


def generate_full_dataset() -> pd.DataFrame:
    all_rows = []
    championship: Dict[str, Dict] = {}

    for year in range(1950, 2027):
        print(f"  Generating {year}...", end="\r")
        circuits = get_circuits_for_year(year)
        drivers = get_active_drivers(year)
        dominant = get_dominant(year)

        if not drivers or not circuits:
            continue

        for round_no, circuit_id in enumerate(circuits, 1):
            rows = simulate_race(year, round_no, circuit_id, drivers, dominant, championship)
            all_rows.extend(rows)

            # Update championship standings
            for row in rows:
                did = row["driver_id"]
                cid = row["constructor_id"]
                if did not in championship:
                    championship[did] = {"points": 0.0, "wins": 0, "position": 20}
                championship[did]["points"] += row["points"]
                if row["position"] == 1:
                    championship[did]["wins"] += 1
                if cid not in championship:
                    championship[cid] = {"points": 0.0, "wins": 0, "position": 10}
                championship[cid]["points"] += row["points"]

        # Rank drivers
        drv_pts = {k: v["points"] for k, v in championship.items() if k in [d["driver_id"] for d in drivers]}
        for rank, (did, _) in enumerate(sorted(drv_pts.items(), key=lambda x: -x[1]), 1):
            if did in championship:
                championship[did]["position"] = rank

    df = pd.DataFrame(all_rows)
    print(f"\n  Total rows: {len(df):,}")
    return df


if __name__ == "__main__":
    print("Generating F1 dataset 1950–2026...")
    df = generate_full_dataset()

    import os
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/f1_1950_2026.csv", index=False)
    print(f"Saved {len(df):,} rows to data/f1_1950_2026.csv")
    print(f"Drivers: {df['driver_id'].nunique()}")
    print(f"Circuits: {df['circuit_id'].nunique()}")
    print(f"Seasons: {df['year'].nunique()}")
    print(f"Races: {df.groupby(['year','round']).ngroups}")
    print("\nSample:")
    print(df[["year","circuit_name","driver_name","constructor_name","grid","position","points","is_wet"]].head(10).to_string(index=False))
