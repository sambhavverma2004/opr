"""
app.py  -  Streamlit dashboard for carbon-aware AI training job scheduling
Run:  streamlit run app.py
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import model as M

st.set_page_config(page_title="Carbon-Aware AI Job Scheduler", layout="wide")

# Fixed colour per region (colour follows the entity, never its rank)
RCOL = {"DE": "#2a78d6", "GB": "#eb6834", "FR": "#1baf7a", "CA": "#eda100"}
BLUE_SEQ = [[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]]
ORANGE_SEQ = [[0, "#f7f6f3"], [0.35, "#f6c3a8"], [1, "#a8401a"]]
LAYOUT = dict(template="plotly_white", font=dict(size=13),
              margin=dict(l=10, r=10, t=40, b=10), hoverlabel=dict(bgcolor="white"))


# ------------------------------------------------------------------ cached helpers
@st.cache_data
def get_carbon():
    return M.load_carbon()


@st.cache_data
def get_jobs():
    return M.load_jobs()


@st.cache_data(show_spinner=False)
def cached_solve(start, horizon, jobs, strategy, cap_tuple, kw, pue):
    win = M.carbon_window(get_carbon(), start, horizon)
    return M.solve(win, jobs, strategy, capacity=dict(cap_tuple), kw_per_gpu=kw, pue=pue)


# ------------------------------------------------------------------ sidebar
ci = get_carbon()
st.sidebar.title("⚙️ Scenario settings")
start_date = st.sidebar.date_input("Window start (UTC, year 2020)", pd.Timestamp(M.DEFAULT_START).date(),
                                   min_value=pd.Timestamp("2020-01-01").date(),
                                   max_value=pd.Timestamp("2020-12-28").date())
start = f"{start_date} 00:00"
horizon = st.sidebar.select_slider("Planning horizon (hours)", [24, 48, 72], value=M.DEFAULT_HORIZON)
st.sidebar.markdown("**Spare GPUs per region (per hour)**")
cap = {}
cols = st.sidebar.columns(2)
for i, r in enumerate(M.REGIONS):
    cap[r] = cols[i % 2].number_input(r, 0, 256, M.DEFAULT_CAPACITY[r], step=4)
kw = st.sidebar.slider("Power per GPU incl. server (kW)", 0.3, 1.5, M.DEFAULT_KW_PER_GPU, 0.05)
pue = st.sidebar.slider("Data-centre PUE", 1.0, 2.0, M.DEFAULT_PUE, 0.02)
show = st.sidebar.radio("Strategy shown in 'Schedule' tab", list(M.STRATEGY_LABELS),
                        index=3, format_func=lambda s: M.STRATEGY_LABELS[s])
st.sidebar.caption("Data: ENTSO-E & CAISO via Wiesner et al. (2021), accessed 19 Nov 2026.")

if "jobs" not in st.session_state:
    st.session_state.jobs = get_jobs()
jobs = st.session_state.jobs
cap_t = tuple(sorted(cap.items()))
win = M.carbon_window(ci, start, horizon)
times = win.index

st.title("🌱 Carbon-Aware Scheduling of AI Training Jobs")
st.caption("Operations Research project · Linear & Integer Programming with Gurobi · "
           "Where and when should GPU jobs run to emit the least CO₂?")

results = {s: cached_solve(start, horizon, jobs, s, cap_t, kw, pue) for s in M.STRATEGY_LABELS}
base, sel = results["baseline"], results[show]

tabs = st.tabs(["📊 Problem & data", "🗓️ Optimal schedule", "⚖️ Compare strategies",
                "💰 Shadow prices", "📈 Sensitivity", "🧮 Formulation"])

# ================================================================== TAB 1
with tabs[0]:
    st.subheader("Grid carbon intensity in the planning window")
    st.markdown("Real hourly data, 2020. Lower = cleaner electricity. "
                "France is mostly nuclear; Germany still burns coal; wind and solar make GB and "
                "California swing during the day.")
    c1, c2, c3, c4 = st.columns(4)
    for col, r in zip([c1, c2, c3, c4], M.REGIONS):
        col.metric(M.REGION_NAMES[r], f"{win[r].mean():.0f} g/kWh")
        col.caption(f"mean · range {win[r].min():.0f}–{win[r].max():.0f} g/kWh")
    fig = go.Figure()
    for r in M.REGIONS:
        fig.add_scatter(x=times, y=win[r], name=M.REGION_NAMES[r], mode="lines",
                        line=dict(color=RCOL[r], width=2),
                        hovertemplate="%{x|%d %b %H:00}<br>%{y:.0f} gCO₂/kWh")
    fig.update_layout(**LAYOUT, height=360, hovermode="x unified",
                      yaxis_title="gCO₂ per kWh", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")

    hm = go.Figure(go.Heatmap(z=win[M.REGIONS].T.values, x=times, y=[M.REGION_NAMES[r] for r in M.REGIONS],
                              colorscale=BLUE_SEQ, colorbar=dict(title="g/kWh"),
                              hovertemplate="%{y}<br>%{x|%d %b %H:00}<br>%{z:.0f} gCO₂/kWh<extra></extra>"))
    hm.update_layout(**LAYOUT, height=230, title="Same data as a heatmap (the optimiser looks for light cells)")
    st.plotly_chart(hm, width="stretch")

    st.subheader("Training jobs to schedule (editable)")
    st.caption("allowed_regions: ANY = all 4 regions, EU = DE/GB/FR (data-protection rules), US = CA only. "
               "release_h / deadline_h are hours from the window start.")
    edited = st.data_editor(jobs, num_rows="dynamic", width="stretch", key="editor",
                            column_config={"allowed_regions": st.column_config.SelectboxColumn(
                                options=list(M.RESIDENCY)),
                                "home_region": st.column_config.SelectboxColumn(options=M.REGIONS)})
    b1, b2 = st.columns([1, 6])
    if b1.button("Apply changes", type="primary"):
        st.session_state.jobs = edited.dropna().reset_index(drop=True)
        st.rerun()
    if b2.button("Reset to default jobs"):
        st.session_state.jobs = get_jobs()
        st.rerun()
    st.caption(f"Total work: **{jobs.gpu_hours.sum():,.0f} GPU-hours** · "
               f"Total capacity in window: **{sum(cap.values()) * horizon:,} GPU-hours**")

# ================================================================== TAB 2
with tabs[1]:
    st.subheader(M.STRATEGY_LABELS[show])
    if sel.co2_kg is None:
        st.error(f"Model is **{sel.status}**. Constraints in conflict (Gurobi IIS):")
        st.code("\n".join(sel.iis) or "n/a")
        st.info("Fix: relax a deadline, raise a capacity, or lower that job's GPU-hours.")
    else:
        saving = None if base.co2_kg is None else base.co2_kg - sel.co2_kg
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Baseline CO₂ (run ASAP at home)", f"{base.co2_kg:,.1f} kg" if base.co2_kg else "infeasible")
        k2.metric("Optimised CO₂", f"{sel.co2_kg:,.1f} kg",
                  None if saving is None else f"-{100 * saving / base.co2_kg:.1f}%", delta_color="inverse")
        k3.metric("CO₂ avoided", "–" if saving is None else f"{saving:,.1f} kg")
        k4.metric("Solver", sel.status)
        k4.caption(f"{sel.solve_seconds:.2f} s · {sel.n_vars} variables · {sel.n_constrs} constraints")

        sch = sel.schedule
        # Job x hour grid coloured by region
        st.markdown("**Job timeline** – each cell is one hour; colour shows the region used")
        job_order = list(jobs.job_id)
        names = dict(zip(jobs.job_id, jobs.job_name))
        g = go.Figure()
        for r in M.REGIONS:
            s = sch[sch.region == r]
            if s.empty:
                continue
            g.add_bar(x=[1] * len(s), base=s.hour, y=s.job_id, orientation="h",
                      name=M.REGION_NAMES[r], marker=dict(color=RCOL[r], line=dict(color="white", width=1)),
                      customdata=list(zip(s.gpu_hours.round(1), s.kg_co2.round(2), s.job_id.map(names))),
                      hovertemplate="%{customdata[2]}<br>hour %{base}<br>%{customdata[0]} GPUs · "
                                    "%{customdata[1]} kg CO₂<extra>" + r + "</extra>")
        for _, jb in jobs.iterrows():   # deadline markers
            g.add_scatter(x=[min(jb.deadline_h, horizon)], y=[jb.job_id], mode="markers",
                          marker=dict(symbol="line-ns", size=16, line=dict(width=2, color="#52514e")),
                          showlegend=False, hovertemplate=f"deadline h{jb.deadline_h}<extra></extra>")
        g.update_layout(**LAYOUT, barmode="overlay", height=60 + 32 * len(job_order),
                        xaxis=dict(title="Hour from window start", range=[0, horizon]),
                        yaxis=dict(categoryorder="array", categoryarray=job_order[::-1]),
                        legend=dict(orientation="h", y=1.08))
        st.plotly_chart(g, width="stretch")
        st.caption("Grey tick = deadline. Jobs move into low-carbon hours and regions.")

        # GPU utilisation per region (small multiples, one axis each)
        st.markdown("**GPU use per region vs. capacity**")
        use = sch.groupby(["region", "hour"]).gpu_hours.sum().unstack(0).reindex(range(horizon)).fillna(0)
        sm = make_subplots(rows=1, cols=4, shared_yaxes=True,
                           subplot_titles=[M.REGION_NAMES[r] for r in M.REGIONS])
        for i, r in enumerate(M.REGIONS, start=1):
            y = use[r] if r in use else [0] * horizon
            sm.add_bar(x=list(range(horizon)), y=y, marker_color=RCOL[r], name=r, showlegend=False,
                       hovertemplate="hour %{x}: %{y:.1f} GPUs<extra>" + r + "</extra>", row=1, col=i)
            sm.add_hline(y=cap[r], line_dash="dot", line_color="#52514e", row=1, col=i)
        sm.update_layout(**LAYOUT, height=280, bargap=0.15)
        st.plotly_chart(sm, width="stretch")
        st.caption("Dotted line = spare-GPU capacity. Full bars mean capacity is binding → non-zero shadow price.")

        st.markdown("**Placement summary (GPU-hours)**")
        pt = sch.pivot_table(index="job_id", columns="region", values="gpu_hours", aggfunc="sum",
                             fill_value=0).reindex(job_order).round(1)
        pt["kg CO₂"] = sch.groupby("job_id").kg_co2.sum().reindex(job_order).round(2)
        pt.insert(0, "job", pt.index.map(names))
        st.dataframe(pt, width="stretch")
        st.download_button("Download schedule CSV", sch.to_csv(index=False), "schedule.csv")

# ================================================================== TAB 3
with tabs[2]:
    st.subheader("How much does each idea save?")
    summ = M.summary_table(results)
    ok = summ.dropna(subset=["CO2 (kg)"])
    bar = go.Figure(go.Bar(x=ok["CO2 (kg)"], y=ok["Strategy"], orientation="h",
                           marker=dict(color="#2a78d6"), text=ok["CO2 (kg)"].map("{:,.1f} kg".format),
                           textposition="outside", hovertemplate="%{y}: %{x:.1f} kg<extra></extra>"))
    bar.update_layout(**LAYOUT, height=300, bargap=0.45, xaxis_title="kg CO₂ for the whole job set",
                      xaxis_range=[0, ok["CO2 (kg)"].max() * 1.18],
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(bar, width="stretch")
    st.dataframe(summ, width="stretch", hide_index=True)
    lp, mip = results["lp"].co2_kg, results["mip"].co2_kg
    if lp and mip:
        st.info(f"**LP vs MIP:** forcing each job into a single region costs "
                f"{mip - lp:.2f} kg extra ({100 * (mip - lp) / lp:.1f}%). The LP is a *relaxation* of the MIP, "
                f"so LP ≤ MIP always. Binary variables: {results['mip'].n_int_vars}. "
                f"MIP gap: {results['mip'].mip_gap:.2%}.")

# ================================================================== TAB 4
with tabs[3]:
    st.subheader("Shadow prices (dual values) from the LP")
    lpres = results["lp"]
    if lpres.capacity_duals is None:
        st.warning("LP is infeasible, so there are no dual values.")
    else:
        st.markdown("**Capacity shadow price** = kg CO₂ that would be saved if **one more GPU-hour** were "
                    "available in that region and hour. Zero means spare capacity (non-binding constraint).")
        cd = lpres.capacity_duals
        h2 = go.Figure(go.Heatmap(z=cd.T.values, x=list(range(horizon)),
                                  y=[M.REGION_NAMES[r] for r in M.REGIONS], colorscale=ORANGE_SEQ,
                                  colorbar=dict(title="kg/GPU-h"),
                                  hovertemplate="%{y}, hour %{x}<br>%{z:.4f} kg CO₂ saved per extra GPU-hour"
                                                "<extra></extra>"))
        h2.update_layout(**LAYOUT, height=250, xaxis_title="Hour from window start")
        st.plotly_chart(h2, width="stretch")
        best = cd.stack().idxmax()
        st.success(f"Most valuable extra GPU: **{M.REGION_NAMES[best[1]]}, hour {best[0]}** → "
                   f"{cd.stack().max():.3f} kg CO₂ saved per GPU-hour. "
                   f"Binding capacity cells: {(cd > 1e-9).sum().sum()} of {cd.size}.")
        st.markdown("**Demand shadow price** = extra kg CO₂ if a job needed **one more GPU-hour** of work.")
        dd = lpres.demand_duals.reindex(jobs.job_id)
        db = go.Figure(go.Bar(x=dd.index, y=dd.values, marker_color="#2a78d6",
                              hovertemplate="%{x}: %{y:.4f} kg per GPU-hour<extra></extra>"))
        db.update_layout(**LAYOUT, height=260, bargap=0.5, yaxis_title="kg CO₂ per extra GPU-hour")
        st.plotly_chart(db, width="stretch")
        st.caption("Jobs restricted to dirtier regions (e.g. US-only data) have higher marginal emissions.")

# ================================================================== TAB 5
with tabs[4]:
    st.subheader("Sensitivity analysis")
    param = st.selectbox("Vary which parameter?", [
        "France (cleanest region) capacity", "All capacities (multiplier)", "Data-centre PUE",
        "Deadline of flexible jobs (h)"])
    if st.button("Run sweep", type="primary"):
        rows = []
        prog = st.progress(0.0)
        if param.startswith("France"):
            vals = [0, 4, 8, 12, 16, 24, 32, 48, 64]
        elif param.startswith("All"):
            vals = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
        elif param.startswith("Data"):
            vals = [1.1, 1.2, 1.3, 1.4, 1.58, 1.8, 2.0]
        else:
            vals = [h for h in [24, 30, 36, 42, 48, 60, 72] if h <= horizon]
        for i, v in enumerate(vals):
            c, p, jb = dict(cap), pue, jobs
            if param.startswith("France"):
                c["FR"] = v
            elif param.startswith("All"):
                c = {k: int(round(x * v)) for k, x in cap.items()}
            elif param.startswith("Data"):
                p = v
            else:
                jb = jobs.assign(deadline_h=jobs.deadline_h.where(jobs.deadline_h < 48, v))
            ct = tuple(sorted(c.items()))
            row = {"value": v}
            for s in ["baseline", "time_shift", "lp", "mip"]:
                row[s] = cached_solve(start, horizon, jb, s, ct, kw, p).co2_kg
            rows.append(row)
            prog.progress((i + 1) / len(vals))
        df = pd.DataFrame(rows)
        f = go.Figure()
        for s, colr in zip(["baseline", "time_shift", "lp", "mip"], ["#52514e", "#eb6834", "#1baf7a", "#2a78d6"]):
            f.add_scatter(x=df.value, y=df[s], name=M.STRATEGY_LABELS[s], mode="lines+markers",
                          line=dict(color=colr, width=2), marker=dict(size=8))
        f.update_layout(**LAYOUT, height=380, xaxis_title=param, yaxis_title="kg CO₂",
                        hovermode="x unified", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(f, width="stretch")
        st.dataframe(df.round(1), width="stretch", hide_index=True)
        st.caption("A gap in a line means that setting is infeasible.")

# ================================================================== TAB 6
with tabs[5]:
    st.subheader("Mathematical model")
    st.markdown("**Sets:** jobs $J$, regions $R=\\{DE,GB,FR,CA\\}$, hours $T$; allowed regions $A_j$; "
                "allowed hours $W_j=\\{t: a_j \\le t < d_j\\}$")
    st.markdown("**Parameters:** work $E_j$ (GPU-h), parallelism $m_j$, capacity $K_r$, "
                "carbon intensity $g_{rt}$ (g/kWh), energy $e$ = kW per GPU × PUE (kWh per GPU-h)")
    st.markdown("**Decision variables:** $x_{jrt} \\ge 0$ GPU-hours of job $j$ in region $r$ at hour $t$; "
                "$y_{jr}\\in\\{0,1\\}$ (MIP only)")
    st.latex(r"\min \; Z = \sum_{j\in J}\sum_{r\in A_j}\sum_{t\in W_j} \frac{e\, g_{rt}}{1000}\, x_{jrt}"
             r"\quad [\text{kg CO}_2]")
    st.latex(r"\sum_{r\in A_j}\sum_{t\in W_j} x_{jrt} = E_j \quad \forall j \qquad \text{(demand)}")
    st.latex(r"\sum_{j} x_{jrt} \le K_r \quad \forall r,t \qquad \text{(capacity)}")
    st.latex(r"\sum_{r} x_{jrt} \le m_j \quad \forall j,t \qquad \text{(parallelism)}")
    st.latex(r"\sum_{r} y_{jr} = 1,\quad \sum_{t} x_{jrt} \le E_j\, y_{jr} \quad \text{(MIP: one region per job)}")
    st.markdown("This is a **transportation problem** (jobs = sources, region-hour slots = destinations, "
                "carbon = unit cost) with blocked cells for deadlines/data residency, extra side constraints, "
                "and a binary extension.")
