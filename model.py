"""
model.py  -  Carbon-aware scheduling of AI training jobs (Gurobi)
=================================================================

SETS
  J : training jobs                         (data/jobs.csv)
  R : data-centre regions {DE, GB, FR, CA}
  T : hours of the planning horizon {0, 1, ..., H-1}
  A_j  subset of R : regions job j may legally use (data-residency rules)
  W_j  subset of T : hours job j may run, release_j <= t < deadline_j

PARAMETERS
  E_j   : GPU-hours of work job j needs
  m_j   : max GPUs job j can use in one hour (parallelism limit)
  K_r   : spare GPUs in region r per hour
  g_rt  : grid carbon intensity of region r in hour t   [gCO2 / kWh]  (real data)
  e     : energy per GPU-hour = kW per GPU x PUE        [kWh / GPU-hour]

DECISION VARIABLES
  x_jrt >= 0 : GPU-hours of job j run in region r during hour t
               (only created for r in A_j and t in W_j  -> "forbidden cells" do not exist)
  y_jr  in {0,1} : 1 if job j is placed in region r          (MIP strategy only)

OBJECTIVE
  min  sum_{j,r,t}  e * g_rt * x_jrt / 1000            [kg CO2]

CONSTRAINTS
  (1) Demand      : sum_{r,t} x_jrt            =  E_j          for every job j
  (2) Capacity    : sum_j     x_jrt            <= K_r          for every region r, hour t
  (3) Parallelism : sum_r     x_jrt            <= m_j          for every job j, hour t
  (4) Bounds      : 0 <= x_jrt <= m_j
  MIP only:
  (5) One region  : sum_r y_jr = 1                             for every job j
  (6) Linking     : sum_t x_jrt <= E_j * y_jr                  for every job j, region r

STRATEGIES (same model, different switches)
  baseline   : home region only, run As-Soon-As-Possible (carbon-unaware, today's default)
  time_shift : home region only, minimise carbon      -> temporal shifting
  lp         : any allowed region, job may be split   -> temporal + spatial shifting (LP)
  mip        : any allowed region, ONE region per job -> temporal + spatial shifting (MIP)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

DATA_DIR = Path(__file__).parent / "data"

REGIONS = ["DE", "GB", "FR", "CA"]
REGION_NAMES = {
    "DE": "Germany (Frankfurt)",
    "GB": "Great Britain (London)",
    "FR": "France (Paris)",
    "CA": "California (US-West)",
}
# Data-residency groups used in jobs.csv -> allowed regions
RESIDENCY = {
    "ANY": ["DE", "GB", "FR", "CA"],
    "EU":  ["DE", "GB", "FR"],   # EU data: EU states + UK (UK has an EU adequacy decision)
    "US":  ["CA"],
}
DEFAULT_CAPACITY = {"DE": 48, "GB": 32, "FR": 16, "CA": 32}   # spare GPUs per hour
DEFAULT_KW_PER_GPU = 0.8   # A100 GPU incl. its share of the server (DGX A100: 6.5 kW / 8 GPUs)
DEFAULT_PUE = 1.2          # data-centre Power Usage Effectiveness (efficient hyperscale DC)
DEFAULT_START = "2020-03-02 00:00"
DEFAULT_HORIZON = 48

STRATEGY_LABELS = {
    "baseline":   "Baseline: run ASAP at home",
    "time_shift": "Time-shifting (home region)",
    "lp":         "Time + space shifting (LP)",
    "mip":        "Time + space, one region per job (MIP)",
}


# ----------------------------------------------------------------------------- data
def load_carbon() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "carbon_intensity_2020_hourly.csv",
                     index_col=0, parse_dates=True)
    inv = {v: k for k, v in REGION_NAMES.items()}
    return df.rename(columns=inv)[REGIONS]


def load_jobs() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "jobs.csv")


def carbon_window(ci: pd.DataFrame, start: str, horizon: int) -> pd.DataFrame:
    """Slice H hours of carbon intensity starting at `start` (UTC)."""
    w = ci.loc[pd.Timestamp(start):].iloc[:horizon]
    if len(w) < horizon:
        raise ValueError("Not enough data after the chosen start date.")
    return w


# ----------------------------------------------------------------------------- result
@dataclass
class Result:
    strategy: str
    status: str
    co2_kg: float | None = None
    schedule: pd.DataFrame | None = None        # rows: job, region, hour, gpu_hours, kg
    capacity_duals: pd.DataFrame | None = None  # hour x region, kg CO2 saved per extra GPU-hour
    demand_duals: pd.Series | None = None       # job -> kg CO2 per extra GPU-hour of work
    n_vars: int = 0
    n_constrs: int = 0
    n_int_vars: int = 0
    solve_seconds: float = 0.0
    mip_gap: float | None = None
    iis: list = field(default_factory=list)     # names of conflicting constraints if infeasible


# ----------------------------------------------------------------------------- model
def solve(ci_win: pd.DataFrame, jobs: pd.DataFrame, strategy: str = "lp",
          capacity: dict | None = None, kw_per_gpu: float = DEFAULT_KW_PER_GPU,
          pue: float = DEFAULT_PUE, time_limit: float = 30.0) -> Result:
    capacity = capacity or DEFAULT_CAPACITY
    e = kw_per_gpu * pue                           # kWh per GPU-hour
    H = len(ci_win)
    T = range(H)
    g = {(r, t): float(ci_win[r].iloc[t]) for r in REGIONS for t in T}

    home_only = strategy in ("baseline", "time_shift")

    # allowed (job, region, hour) cells
    cells = []
    for _, jb in jobs.iterrows():
        regs = [jb.home_region] if home_only else RESIDENCY[jb.allowed_regions]
        t0, t1 = int(jb.release_h), min(int(jb.deadline_h), H)
        for r in regs:
            for t in range(t0, t1):
                cells.append((jb.job_id, r, t))

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("carbon_aware_scheduling", env=env)
    m.Params.TimeLimit = time_limit

    maxg = dict(zip(jobs.job_id, jobs.max_gpus))
    work = dict(zip(jobs.job_id, jobs.gpu_hours))

    x = m.addVars(cells, lb=0.0, ub={c: maxg[c[0]] for c in cells}, name="x")

    # (1) demand
    dem = {j: m.addConstr(x.sum(j, "*", "*") == work[j], name=f"demand[{j}]")
           for j in jobs.job_id}
    # (2) capacity
    cap = {(r, t): m.addConstr(x.sum("*", r, t) <= capacity[r], name=f"capacity[{r},{t}]")
           for r in REGIONS for t in T}
    # (3) parallelism across regions in the same hour
    for j in jobs.job_id:
        for t in T:
            if any(True for _ in x.select(j, "*", t)):
                m.addConstr(x.sum(j, "*", t) <= maxg[j], name=f"parallel[{j},{t}]")

    if strategy == "mip":
        jr = sorted({(c[0], c[1]) for c in cells})
        y = m.addVars(jr, vtype=GRB.BINARY, name="y")
        for j in jobs.job_id:
            m.addConstr(y.sum(j, "*") == 1, name=f"one_region[{j}]")
        for (j, r) in jr:
            m.addConstr(x.sum(j, r, "*") <= work[j] * y[j, r], name=f"link[{j},{r}]")

    carbon_expr = gp.quicksum(e * g[r, t] / 1000.0 * x[j, r, t] for (j, r, t) in cells)
    if strategy == "baseline":
        # carbon-unaware: finish as early as possible (weight = hour index)
        m.setObjective(gp.quicksum((t + 1) * x[j, r, t] for (j, r, t) in cells), GRB.MINIMIZE)
    else:
        m.setObjective(carbon_expr, GRB.MINIMIZE)

    t_start = time.time()
    m.optimize()
    res = Result(strategy=strategy, status="", n_vars=m.NumVars,
                 n_constrs=m.NumConstrs, n_int_vars=m.NumIntVars,
                 solve_seconds=time.time() - t_start)

    if m.Status == GRB.INFEASIBLE:
        res.status = "INFEASIBLE"
        m.computeIIS()
        res.iis = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
        env.dispose()
        return res
    if m.SolCount == 0:
        res.status = f"NO SOLUTION (status {m.Status})"
        env.dispose()
        return res

    res.status = "OPTIMAL" if m.Status == GRB.OPTIMAL else f"STATUS {m.Status}"
    rows = [(j, r, t, x[j, r, t].X, e * g[r, t] / 1000.0 * x[j, r, t].X)
            for (j, r, t) in cells if x[j, r, t].X > 1e-6]
    res.schedule = pd.DataFrame(rows, columns=["job_id", "region", "hour", "gpu_hours", "kg_co2"])
    res.co2_kg = float(res.schedule.kg_co2.sum())
    if strategy == "mip":
        res.mip_gap = m.MIPGap

    # Shadow prices only exist for continuous models (LP)
    if strategy in ("lp", "time_shift"):
        # Pi of a <= constraint in a min problem is <= 0. -Pi = kg CO2 saved per extra GPU-hour.
        res.capacity_duals = pd.DataFrame(
            [[-cap[r, t].Pi + 0.0 for r in REGIONS] for t in T], columns=REGIONS)
        res.demand_duals = pd.Series({j: dem[j].Pi for j in jobs.job_id})

    env.dispose()
    return res


def run_all(ci_win, jobs, **kw) -> dict[str, Result]:
    return {s: solve(ci_win, jobs, strategy=s, **kw) for s in STRATEGY_LABELS}


def summary_table(results: dict[str, Result]) -> pd.DataFrame:
    base = results["baseline"].co2_kg
    rows = []
    for s, r in results.items():
        saved = None if (r.co2_kg is None or base is None) else 100 * (base - r.co2_kg) / base
        rows.append({"Strategy": STRATEGY_LABELS[s], "Status": r.status,
                     "CO2 (kg)": None if r.co2_kg is None else round(r.co2_kg, 1),
                     "Saved vs baseline (%)": None if saved is None else round(saved, 1),
                     "Variables": r.n_vars, "Integer vars": r.n_int_vars,
                     "Constraints": r.n_constrs, "Solve time (s)": round(r.solve_seconds, 3)})
    return pd.DataFrame(rows)
