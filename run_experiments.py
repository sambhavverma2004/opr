"""
run_experiments.py  -  reproduces every number in REPORT.md
Usage:  python run_experiments.py
Writes CSV files to results/.
"""
from pathlib import Path

import pandas as pd

import model as M

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 160)


def main():
    ci, jobs = M.load_carbon(), M.load_jobs()
    win = M.carbon_window(ci, M.DEFAULT_START, M.DEFAULT_HORIZON)

    print(f"\n=== Base case: {M.DEFAULT_START} UTC, {M.DEFAULT_HORIZON} h ===")
    print("Mean carbon intensity in window (gCO2/kWh):")
    print(win.mean().round(1).to_string())
    res = M.run_all(win, jobs)
    summ = M.summary_table(res)
    print("\n", summ.to_string(index=False))
    summ.to_csv(OUT / "summary_base_case.csv", index=False)

    # Where did each job go? (MIP)
    mip = res["mip"].schedule
    place = (mip.groupby(["job_id", "region"]).gpu_hours.sum().round(1)
             .unstack(fill_value=0))
    print("\nMIP placement (GPU-hours per region):\n", place.to_string())
    mip.to_csv(OUT / "schedule_mip.csv", index=False)
    res["lp"].schedule.to_csv(OUT / "schedule_lp.csv", index=False)

    # Shadow prices (LP)
    cd = res["lp"].capacity_duals
    print("\nLP capacity shadow prices (kg CO2 saved per extra GPU-hour), mean over hours:")
    print(cd.mean().round(4).to_string())
    print("Hours where capacity is binding (dual > 0):", int((cd > 1e-9).sum().sum()), "of", cd.size)
    cd.to_csv(OUT / "capacity_shadow_prices_lp.csv", index_label="hour")
    dd = res["lp"].demand_duals.round(4)
    print("\nLP demand shadow prices (kg CO2 per extra GPU-hour of work):\n", dd.to_string())
    dd.to_csv(OUT / "demand_shadow_prices_lp.csv", header=["kg_per_gpu_hour"])

    # Scenario analysis
    print("\n=== Scenarios (MIP) ===")
    scen = []
    base_mip = res["mip"].co2_kg
    for name, kwargs, jb in [
        ("Base case", {}, jobs),
        ("France capacity halved (16->8)", {"capacity": {**M.DEFAULT_CAPACITY, "FR": 8}}, jobs),
        ("All capacities +50%", {"capacity": {k: int(v * 1.5) for k, v in M.DEFAULT_CAPACITY.items()}}, jobs),
        ("48-h deadlines cut to 36 h", {}, jobs.assign(deadline_h=jobs.deadline_h.clip(upper=36))),
        ("All jobs EU-only (stricter residency)", {}, jobs.assign(allowed_regions=jobs.allowed_regions.replace("ANY", "EU"))),
        ("PUE 1.58 (industry average DC)", {"pue": 1.58}, jobs),
    ]:
        r = M.solve(win, jb, "mip", **kwargs)
        b = M.solve(win, jb, "baseline", **kwargs)
        scen.append({"Scenario": name, "Status": r.status,
                     "Baseline CO2 (kg)": None if b.co2_kg is None else round(b.co2_kg, 1),
                     "MIP CO2 (kg)": None if r.co2_kg is None else round(r.co2_kg, 1),
                     "Change vs base MIP (%)": None if r.co2_kg is None else round(100 * (r.co2_kg - base_mip) / base_mip, 1),
                     "IIS (if infeasible)": "; ".join(r.iis[:6])})
    scen = pd.DataFrame(scen)
    print(scen.to_string(index=False))
    scen.to_csv(OUT / "scenarios.csv", index=False)

    # Infeasibility demo
    print("\n=== Infeasibility demo: Nightly retrain (J3) must finish in 6 h ===")
    r = M.solve(win, jobs.assign(deadline_h=jobs.deadline_h.where(jobs.job_id != "J3", 6)), "lp")
    print(r.status, "| IIS:", r.iis)

    # Robustness: every Monday of 2020
    print("\n=== Robustness: 51 weekly windows in 2020 ===")
    rows = []
    for d in pd.date_range("2020-01-06", "2020-12-21", freq="7D"):
        w = M.carbon_window(ci, str(d), M.DEFAULT_HORIZON)
        rr = M.run_all(w, jobs)
        b = rr["baseline"].co2_kg
        rows.append({"start": d.date(), "baseline_kg": b,
                     **{f"{s}_saving_pct": 100 * (b - rr[s].co2_kg) / b for s in ["time_shift", "lp", "mip"]}})
    yr = pd.DataFrame(rows)
    yr.to_csv(OUT / "year_2020_weekly.csv", index=False)
    print(yr.describe().loc[["mean", "min", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
