from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Tuple

DATA_DIR_DEFAULT = "../data"
OUT_DIR_DEFAULT = "./images/plotly"

START_TS = datetime(2020, 4, 6, 0, 0, 0)
END_TS = datetime(2020, 4, 10, 23, 59, 59)

NIGHT_INTERVALS: List[Tuple[datetime, datetime]] = [
    (datetime(2020, 4, 6, 0, 0, 0), datetime(2020, 4, 6, 6, 0, 0)),
    (datetime(2020, 4, 6, 18, 0, 0), datetime(2020, 4, 7, 6, 0, 0)),
    (datetime(2020, 4, 7, 18, 0, 0), datetime(2020, 4, 8, 6, 0, 0)),
    (datetime(2020, 4, 8, 18, 0, 0), datetime(2020, 4, 9, 6, 0, 0)),
    (datetime(2020, 4, 9, 18, 0, 0), datetime(2020, 4, 10, 6, 0, 0)),
    (datetime(2020, 4, 10, 18, 0, 0), datetime(2020, 4, 10, 23, 59, 59)),
]

# From MC2_datadescription.docx (lat, lon)
HOSPITALS_LATLON = [
    (0.180960, -119.959400),
    (0.153120, -119.915900),
    (0.151090, -119.909520),
    (0.121800, -119.904300),
    (0.134560, -119.883420),
    (0.182990, -119.855580),
    (0.041470, -119.828610),
    (0.065250, -119.744800),
]
NUCLEAR_PLANT_LATLON = (0.162679, -119.784825)

LON_MIN = min([lon for lat, lon in HOSPITALS_LATLON] + [NUCLEAR_PLANT_LATLON[1]]) - 0.05
LON_MAX = max([lon for lat, lon in HOSPITALS_LATLON] + [NUCLEAR_PLANT_LATLON[1]]) + 0.05
LAT_MIN = min([lat for lat, lon in HOSPITALS_LATLON] + [NUCLEAR_PLANT_LATLON[0]]) - 0.05
LAT_MAX = max([lat for lat, lon in HOSPITALS_LATLON] + [NUCLEAR_PLANT_LATLON[0]]) + 0.05


@dataclass(frozen=True)
class Poi:
    kind: str
    label: str
    lon: float
    lat: float


def iter_pois() -> Iterable[Poi]:
    for lat, lon in HOSPITALS_LATLON:
        yield Poi(kind="hospital", label="Ⓗ", lon=lon, lat=lat)
    lat, lon = NUCLEAR_PLANT_LATLON
    yield Poi(kind="nuclear_plant", label="☢️", lon=lon, lat=lat)