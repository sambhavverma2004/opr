"""
prepare_data.py
---------------
Builds data/carbon_intensity_2020_hourly.csv from the public "Let's Wait Awhile"
dataset (Wiesner et al., ACM Middleware 2021).

Source repository : https://github.com/dos-group/lets-wait-awhile  (folder data/)
Original sources  : ENTSO-E Transparency Platform (Germany, Great Britain, France)
                    California ISO (California)
Accessed          : 19 November 2026

You only need to run this if you want to rebuild the CSV yourself.
The processed CSV is already included in data/.

    git clone --depth 1 https://github.com/dos-group/lets-wait-awhile.git
    python prepare_data.py lets-wait-awhile/data

What it does
  1. Reads each region's carbon intensity file (gCO2/kWh, 5-30 min resolution).
  2. California timestamps are in local Pacific time (its solar peak is at ~11:00),
     so they are converted to UTC. The European files are already in UTC.
  3. Averages everything to 1-hour resolution and keeps calendar year 2020.
"""
import sys
from pathlib import Path

import pandas as pd

# file name -> (label used in this project, timezone of the raw timestamps)
REGIONS = {
    "ger_ci.csv": ("Germany (Frankfurt)", "UTC"),
    "gb_ci.csv":  ("Great Britain (London)", "UTC"),
    "fr_ci.csv":  ("France (Paris)", "UTC"),
    "cal_ci.csv": ("California (US-West)", "America/Los_Angeles"),
}


def load_region(path: Path, tz: str) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True)["Carbon Intensity"]
    if tz != "UTC":
        s.index = s.index.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
        s = s[s.index.notna()]
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    return s.resample("1h").mean()


def main(raw_dir: str) -> None:
    raw = Path(raw_dir)
    cols = {}
    for fname, (label, tz) in REGIONS.items():
        cols[label] = load_region(raw / fname, tz)
    df = pd.DataFrame(cols)
    df = df.loc["2020-01-01":"2020-12-31 23:00"].interpolate(limit=3).round(2)
    df.index.name = "time_utc"
    out = Path(__file__).parent / "data" / "carbon_intensity_2020_hourly.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out)
    print(f"Wrote {out}  shape={df.shape}  missing={int(df.isna().sum().sum())}")
    print(df.describe().round(1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lets-wait-awhile/data")
