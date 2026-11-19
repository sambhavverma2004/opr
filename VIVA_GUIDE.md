# Viva preparation guide

## Explaining it in 30 seconds
"We schedule 10 AI training jobs across 4 cloud regions over 48 hours to minimise CO₂. Grid carbon
intensity changes by region and hour, and we use real 2020 data from ENTSO-E and California ISO. The
decision variable is how many GPU-hours of each job run in each region in each hour. The model is a
transportation-style LP with capacity, deadline, parallelism and data-residency constraints. A MIP
version forces each job into one region. Compared with running everything immediately in the home
region, we cut CO₂ by about 62 %, and only by about 16 % if jobs are only delayed and never moved."

## Suggested split (but everyone must know everything)
* **Member 1:** problem, data, assumptions (Report §1–2.2; dashboard tab 1)
* **Member 2:** formulation, LP vs MIP, link to transportation problem (§2.3, §3; tabs 3, 6)
* **Member 3:** results, shadow prices, sensitivity, live demo (§4–5; tabs 2, 4, 5)

## How to read the code in 1 hour
1. `data/jobs.csv`: the 10 jobs.
2. `model.py`: read the docstring at the top (it is the whole formulation). Then read `solve()` top to
   bottom: create the cells → `addVars` → constraints (1)(2)(3) → MIP constraints (5)(6) → objective → optimise → read `.X` and `.Pi`.
3. `run_experiments.py`: calls `solve()` for each strategy and scenario.
4. `app.py`: only plotting; no OR logic.

## Likely questions and answers

**Q. What type of problem is this?**
A linear program with a transportation structure. Jobs are supply nodes (supply = GPU-hours needed),
region-hour slots are demand nodes (capacity = spare GPUs), and the unit cost is the CO₂ per GPU-hour.
The MIP version adds binary variables y_jr.

**Q. Why is it not a pure transportation problem?**
It has extra side constraints: the parallelism limit per job per hour (Σ_r x_jrt ≤ m_j), the
one-region rule in the MIP, and blocked cells for deadlines and residency. Also, supply ≠ total
capacity: there is far more capacity (6,144 GPU-h) than work (1,128). In the transportation-method
terms from class, this is an unbalanced problem, which you would balance with a dummy job holding
the unused capacity. Our ≤ capacity constraints play the role of that dummy.

**Q. How did you handle deadlines and data-residency rules?**
As prohibited routes. In the transportation tableau you would put a cost of big-M in those cells. In
code we simply do not create those variables, which is equivalent and makes the model smaller.

**Q. Number of variables and constraints?**
LP: 1,224 variables, 556 constraints (10 demand, 192 capacity, 354 parallelism). MIP: +34 binary
variables and +44 constraints (10 one-region, 34 linking).

**Q. Why is LP ≤ MIP?**
The LP is the *LP relaxation* of the MIP: it removes the integrality (one-region) restriction, so its
feasible region is larger and its minimum can only be lower or equal. Here it is 115.6 vs 118.5 kg.

**Q. How does Gurobi solve the MIP?**
Branch and bound. It solves the LP relaxation, branches on fractional y variables, uses bounds to
prune branches, and adds cutting planes. The MIP gap is (incumbent − best bound)/incumbent; ours is 0 %,
so the solution is proven optimal.

**Q. What is a shadow price here? Give a number.**
The change in the optimal objective per unit increase in a constraint's right-hand side.
* Capacity in France at hour 0: 0.138 kg CO₂ saved per extra GPU-hour.
* Demand of job J1: 0.175 kg per extra GPU-hour of work, i.e. 0.175/0.96 ≈ 182 g/kWh. That is the
  carbon intensity of the "marginal" slot that would absorb the extra work.

**Q. Why are Germany's capacity shadow prices zero?**
Germany's capacity is never fully used, because it is the dirtiest region. A non-binding constraint
has a zero dual (**complementary slackness**).

**Q. Why does J4 have a higher demand dual?**
It holds US health data, so it can only run in California. Its marginal slot is dirtier (≈301 g/kWh).

**Q. Relation between the duals and the MODI method?**
In MODI, u_i (rows) and v_j (columns) are exactly the dual variables of the supply and demand
constraints. Here the demand duals play the role of u_i and the capacity duals the role of v_j.
The optimality test c_ij − u_i − v_j ≥ 0 corresponds to reduced cost ≥ 0 in the simplex method.

**Q. What happens if the problem is infeasible? Show an example.**
If J3 must finish within 6 h, it can do at most 8 GPUs × 6 h = 48 < 64 GPU-h. Gurobi's IIS reports
the conflicting set: `demand[J3]` + `parallel[J3,0..5]`. You can show this live by setting the
horizon to 24 h in the dashboard: J10 starts at hour 24, so it has no feasible slot.

**Q. Is the data real?**
The carbon intensity is real hourly 2020 data (ENTSO-E and California ISO via Wiesner et al. 2021).
The job list and capacities are assumptions. They are typical ML workloads, and they are editable in the dashboard.

**Q. How is CO₂ computed?**
kg = GPU-hours × 0.8 kW × PUE 1.2 × g/kWh ÷ 1000. 0.8 kW per GPU comes from the DGX A100's 6.5 kW across 8 GPUs.

**Q. Why does time-shifting save less than moving regions?**
Within one region, intensity varies by perhaps 2× over a day. Between regions it varies by about 6× (France 70 vs Germany 399).

**Q. Limitations?**
Perfect forecasts; jobs assumed pre-emptible; network energy ignored; average rather than marginal
emission factors; electricity cost not included. Extensions: stochastic or robust optimisation,
multi-objective (cost + CO₂), binary start times for non-pre-emptible jobs.

**Q. Why not just send everything to France?**
France has only 16 spare GPUs per hour (768 GPU-h in 48 h), but the jobs need 1,128 GPU-h. Some
jobs are also restricted (J4 must stay in the US). The capacity constraint is binding, which is
exactly what its non-zero shadow price shows.

## Live demo script (3 minutes)
1. Tab **Problem & data**: point at the heatmap. France is light (clean), Germany dark.
2. Tab **Optimal schedule**: timeline. Most jobs are green (France); J4, J6 and J9 are yellow (California), mostly in its cleaner solar hours.
3. Tab **Compare strategies**: 306.6 → 257.2 → 115.6 / 118.5 kg.
4. Tab **Shadow prices**: the France row is orange (binding); the others are white.
5. Sidebar: set FR capacity to 8, and CO₂ rises. Set the horizon to 24, and you get an infeasible result with the IIS.
6. Tab **Sensitivity**: run the "France capacity" sweep and show diminishing returns.
