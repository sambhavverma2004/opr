# Carbon-Aware Scheduling of AI Training Jobs across Data-Centre Regions
**Operations Research – Course Project**
Group members: ______________________ · ______________________ · ______________________
Methods used: **Linear Programming (simplex), Transportation-problem structure, Integer Programming** · Solver: **Gurobi**

---

## 1. Problem statement

Training machine-learning models uses large amounts of electricity. How much CO₂ that electricity
causes depends on **where** and **when** it is used. In March 2020, the French grid (mostly nuclear)
emitted about 70 gCO₂/kWh, while the German grid (coal) emitted about 400 gCO₂/kWh. Within one
region, intensity also changes hour by hour with wind and solar output.

Most training jobs are **flexible**: a model must be ready by a deadline, but it does not matter
at which hour it trains. Cloud providers have data centres in several regions. This gives two
levers:
* **temporal shifting:** run a job in a cleaner hour;
* **spatial shifting:** run a job in a cleaner region.

**Problem.** A company has 10 GPU training jobs to finish within a 48-hour window. It has spare GPUs
in 4 regions (Frankfurt, London, Paris, California). Decide **how many GPU-hours of each job to run in
each region in each hour** to **minimise total CO₂ emissions**, subject to:
* every job gets all the GPU-hours it needs, between its release time and its deadline;
* no region uses more GPUs in an hour than it has spare;
* a job cannot use more GPUs at once than its code can parallelise;
* jobs with personal data may only run in permitted jurisdictions (EU/UK data stays in Europe;
  US health data stays in the US);
* (integer version) each job runs in **one** region only, because moving checkpoints and datasets
  between continents is impractical.

## 2. Data source, assumptions and mathematical formulation

### 2.1 Data
| Data | Source | Access date |
|---|---|---|
| Hourly grid carbon intensity (gCO₂/kWh), Germany, Great Britain, France, California, all of 2020 | Wiesner et al. (2021), *Let's Wait Awhile*, ACM Middleware '21. Dataset: github.com/dos-group/lets-wait-awhile, derived from the **ENTSO-E Transparency Platform** and **California ISO** | 19 Nov 2026 |
| GPU server power: DGX A100 = 8 GPUs, 6.5 kW max → ≈0.8 kW per GPU | NVIDIA DGX A100 datasheet | 19 Nov 2026 |
| Data-centre PUE (industry average 1.58 in 2020; efficient hyperscale ≈1.1–1.2) | Uptime Institute Global Data Center Survey | 19 Nov 2026 |
| Job list (10 jobs) and spare capacity per region | Assumed. Representative of typical ML workloads (see 2.2) | – |

Pre-processing (`prepare_data.py`): 5–30-minute values averaged to hourly. California timestamps
are local Pacific time (solar peak at 11:00), so they were converted to UTC; the European series
are already UTC. Result: 8,784 hourly values per region with no missing values.

**Base-case window:** 2–3 March 2020 (48 h, UTC). Mean intensity: DE 399, GB 259, FR 70, CA 273 gCO₂/kWh.

### 2.2 Assumptions
1. Carbon intensity is known for the next 48 h. In practice, grid operators publish forecasts; we use actual values.
2. Jobs are **pre-emptible** and **divisible**: work can be paused and resumed (checkpointing), and
   GPU-hours are additive (a 64 GPU-h job can be 8 GPUs × 8 h or 4 GPUs × 16 h).
3. Energy per GPU-hour e = 0.8 kW × PUE 1.2 = **0.96 kWh**. This is the same for all regions.
4. Each region has a fixed number of spare GPUs per hour: DE 48, GB 32, FR 16, CA 32.
5. Network transfer energy and latency are ignored. The MIP's one-region rule is a proxy for them.
6. Emissions use average (location-based) grid carbon intensity.
7. The jobs (`data/jobs.csv`) total 1,128 GPU-hours ≈ 1,083 kWh.

| Job | Description | Home | GPU-h | Max GPUs | Window (h) | Allowed |
|---|---|---|---|---|---|---|
| J1 | LLM-7B LoRA fine-tuning | DE | 256 | 16 | 0–48 | EU |
| J2 | ResNet-50 ImageNet training | GB | 192 | 8 | 0–48 | ANY |
| J3 | Nightly recommender retrain | DE | 64 | 8 | 0–12 | EU |
| J4 | Whisper fine-tune (US health data) | CA | 96 | 8 | 0–36 | US |
| J5 | YOLOv8 detection training | CA | 120 | 8 | 6–48 | ANY |
| J6 | Hyper-parameter sweep | GB | 160 | 16 | 0–48 | ANY |
| J7 | Stable Diffusion LoRA | FR | 48 | 4 | 0–24 | ANY |
| J8 | GNN fraud model (EU bank data) | DE | 80 | 8 | 12–36 | EU |
| J9 | PPO RL agent | GB | 72 | 4 | 0–48 | ANY |
| J10 | Batch inference / evaluation | CA | 40 | 8 | 24–48 | ANY |

EU = {DE, GB, FR} (the UK has an EU data-adequacy decision). US = {CA}. ANY = all four.

### 2.3 Mathematical formulation

**Sets**
* J = jobs, R = {DE, GB, FR, CA}, T = {0,…,47} hours
* A_j ⊆ R = regions allowed for job j (data residency)
* W_j = {t : a_j ≤ t < d_j} = hours allowed for job j (release a_j, deadline d_j)

**Parameters**
* E_j = GPU-hours required by job j
* m_j = maximum GPUs job j can use in one hour
* K_r = spare GPUs in region r per hour
* g_rt = carbon intensity of region r in hour t (gCO₂/kWh), real data
* e = kWh per GPU-hour (= kW per GPU × PUE)

**Decision variables**
* x_jrt ≥ 0 = GPU-hours of job j executed in region r during hour t (defined only for r ∈ A_j, t ∈ W_j)
* y_jr ∈ {0,1} = 1 if job j is assigned to region r (integer model only)

**Objective** (kg CO₂)

  min Z = Σ_j Σ_{r∈A_j} Σ_{t∈W_j} (e · g_rt / 1000) · x_jrt

**Constraints**
1. Demand: Σ_{r∈A_j} Σ_{t∈W_j} x_jrt = E_j ∀ j
2. Capacity: Σ_j x_jrt ≤ K_r ∀ r, t
3. Parallelism: Σ_r x_jrt ≤ m_j ∀ j, t   and   0 ≤ x_jrt ≤ m_j
4. *(MIP)* One region: Σ_r y_jr = 1 ∀ j
5. *(MIP)* Linking: Σ_t x_jrt ≤ E_j · y_jr ∀ j, r

**Link to course topics.** Without constraints 3–5 this is exactly a **transportation problem**.
Jobs are sources with supply E_j, (region, hour) slots are destinations with capacity K_r, and the
unit cost is e·g_rt. Deadlines and data-residency rules are **prohibited routes** (like the "M"
cells in the transportation tableau). We do not create those variables at all, which has the same
effect as a big-M cost. Constraints 4–5 turn the LP into a **mixed-integer program**.

## 3. Solution approach

* Implemented in Python with **gurobipy** (`model.py`). The same model is solved under four strategies:

| Strategy | Allowed regions | Objective | Type |
|---|---|---|---|
| Baseline (today's default) | home region only | finish ASAP, min Σ (t+1)·x | LP, then CO₂ evaluated |
| Time-shifting | home region only | min CO₂ | LP |
| Time + space shifting | all allowed regions, jobs may split | min CO₂ | LP (simplex) |
| Time + space, one region | all allowed regions, one per job | min CO₂ | MIP (branch & bound) |

* LPs are solved with Gurobi's simplex/barrier. The MIP uses Gurobi's branch-and-bound with cutting planes.
* **Dual values (shadow prices)** of the capacity and demand constraints are read from the LP.
* **Infeasible instances** are diagnosed with Gurobi's **IIS** (Irreducible Inconsistent Subsystem).
* **Sensitivity / scenario analysis:** capacity, PUE, deadlines, data-residency rules.
* **Robustness:** the whole experiment is repeated for 51 weekly windows covering 2020.
* An interactive **Streamlit dashboard** (`app.py`) re-solves the model live when inputs change.

Model size (base case): LP 1,224 variables and 556 constraints; MIP 1,258 variables (34 binary)
and 600 constraints. Every model solves in under 0.5 s.

## 4. Results

### 4.1 Base case (2–3 March 2020)
| Strategy | CO₂ (kg) | Saving vs baseline |
|---|---|---|
| Baseline: run ASAP at home | **306.6** | – |
| Time-shifting only | 257.2 | 16.1 % |
| Time + space shifting (LP) | **115.6** | **62.3 %** |
| Time + space, one region per job (MIP) | 118.5 | 61.3 % |

* Time-shifting alone saves 16 %. Adding region choice saves 62 %, because France's grid is about 5× cleaner than Germany's.
* **LP vs MIP:** the single-region rule costs only **2.87 kg (+2.5 %)**. Since the LP is a
  relaxation of the MIP, LP ≤ MIP must hold, and it does. MIP gap = 0 %.
* MIP placement: France 760 GPU-h (J1, J2, J3, J5, J7, J8), California 328 GPU-h (J4, J6, J9),
  Great Britain 40 GPU-h (J10), Germany 0. France's capacity (16 × 48 = 768) is almost fully used.

### 4.2 Shadow prices (LP)
* **Demand duals:** 0.175 kg CO₂ per extra GPU-hour for most jobs. Divided by e = 0.96 kWh, this
  equals a **marginal carbon intensity of ≈182 g/kWh**: the cleanest hour still free in GB/CA.
  Job J4 (US-only data) has a dual of 0.289 kg (≈301 g/kWh), because it can only use California.
* **Capacity duals:** binding in 56 of 192 (region, hour) cells, almost all in **France**. In France
  at hour 0 the dual is 0.138 kg per GPU-hour. This equals 0.175 − 0.96 × 38.2/1000: the saving from
  moving one GPU-hour from the marginal slot (182 g/kWh) to France (38 g/kWh). Germany's capacity
  duals are 0, so extra German GPUs have no value. This tells the company **where to buy more capacity**.

### 4.3 Scenario analysis (MIP)
| Scenario | Baseline (kg) | Optimised (kg) | Change vs base MIP |
|---|---|---|---|
| Base case | 306.6 | 118.5 | – |
| France capacity halved (16 → 8) | 306.6 | 164.0 | +38.4 % |
| All capacities +50 % | 306.6 | 87.5 | −26.1 % |
| 48-h deadlines cut to 36 h | 306.6 | 141.7 | +19.5 % |
| All jobs EU-only (stricter residency) | 306.6 | 119.7 | +1.0 % |
| PUE 1.58 (industry-average DC) | 403.7 | 156.0 | +31.7 % |

**Infeasibility demo:** if the nightly retrain J3 (64 GPU-h, max 8 GPUs) must finish within 6 h, the
model is infeasible: at most 8 × 6 = 48 < 64 GPU-h can be done. Gurobi's IIS correctly returns
`demand[J3]` and `parallel[J3,0..5]`.

### 4.4 Robustness over all of 2020 (51 weekly 48-h windows)
| | Time-shift saving | LP saving | MIP saving |
|---|---|---|---|
| Mean | 17.0 % | 63.8 % | 62.9 % |
| Min | 6.9 % | 50.6 % | 50.0 % |
| Max | 30.6 % | 75.0 % | 74.1 % |

The savings are not specific to the chosen week: carbon-aware placement saves at least 50 % in every week of 2020.

## 5. Conclusions

1. A simple LP (a transportation problem with side constraints) cuts the CO₂ of a realistic
   ML workload by **≈62 %** compared with the default "run now, at home" policy.
2. **Where** matters more than **when**: spatial shifting adds about 45 percentage points on top of time-shifting.
3. Requiring each job to stay in one region (MIP) reduces the saving by only **≈1 percentage point**
   on average (62.9 % vs 63.8 %), so it is a practical rule with little environmental cost.
4. Shadow prices identify the bottleneck: clean capacity in France. Extra GPUs there are worth
   ≈0.14 kg CO₂ per GPU-hour; extra GPUs in Germany are worth nothing.
5. Tighter deadlines and less clean capacity reduce the benefit. Data-residency rules cost little
   here, because Europe already contains the cleanest region.

**Limitations and future work:** use carbon-intensity *forecasts* (with uncertainty, i.e. stochastic
or robust optimisation); include electricity price for a cost–carbon multi-objective model; model
network transfer energy; use marginal instead of average emission factors; make jobs
non-pre-emptible (requires binary start-time variables).

## References
1. P. Wiesner, I. Behnke, D. Scheinert, K. Gontarska, L. Thamsen. *Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud.* ACM Middleware 2021. doi:10.1145/3464298.3493399. Data: https://github.com/dos-group/lets-wait-awhile (accessed 19 Nov 2026).
2. ENTSO-E Transparency Platform, https://transparency.entsoe.eu ; California ISO, https://www.caiso.com (original sources of ref. 1).
3. NVIDIA DGX A100 Datasheet (system power 6.5 kW max, 8 × A100).
4. Uptime Institute, Global Data Center Survey 2020 (average PUE 1.58).
5. Gurobi Optimization, LLC. Gurobi Optimizer Reference Manual. Libraries: gurobipy, pandas, Streamlit, Plotly.
6. AI tool: Claude (Anthropic) was used to help write code and documentation. All results come from running the code.
