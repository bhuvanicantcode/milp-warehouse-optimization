# 🗺️ MILP Warehouse Optimization — India Logistics Network

> **BITS Pilani | SCM Portfolio Project** — Pallapolu Bhuvan Chandra  
> Mixed-Integer Linear Programming for warehouse location & demand allocation across 20 Indian cities

---

## 📌 Problem Statement

An FMCG manufacturer must decide **which warehouses to open** (out of 6 candidate Distribution Centres) and **which cities each DC should serve** — minimising the sum of:

- **Fixed annual operating costs** per open warehouse
- **Variable transport costs** (distance × demand × cost-per-km-per-unit)

This is a classic **Facility Location Problem**, solved exactly using MILP (via PuLP / CBC solver).

---

## 🚀 Live Dashboard Features

| Tab | What it shows |
|-----|---------------|
| 🗺️ **Network Map** | Optimal DC locations + city assignments on a India map |
| 📊 **Cost Analysis** | Fixed vs. transport cost per DC, delivery distance distribution |
| 🎯 **Baseline Comparison** | MILP optimal vs. naive (all-open) vs. single-hub strategy |
| ⚡ **Sensitivity Analysis** | Fuel cost shocks (+10–50%) and demand surges (80–150%) |

### Key KPIs computed
- Total annual cost (₹ Crore)
- Cost per unit delivered (₹)
- Demand-weighted average delivery distance (km)
- % demand served within 500 km
- Savings vs. naive and single-hub baselines

---

## 🏗️ Network Data

**6 candidate warehouses** (potential DCs):
Mumbai · Delhi · Bengaluru · Chennai · Kolkata · Hyderabad

**20 demand cities** served (including):
Pune · Ahmedabad · Jaipur · Lucknow · Surat · Kochi · Guwahati · Chandigarh · and more

Distances computed via the **Haversine formula** (great-circle, km).

---

## ⚙️ Installation & Usage

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/milp-warehouse-optimization.git
cd milp-warehouse-optimization
```

### 2. Set up a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit dashboard
```bash
streamlit run milp_warehouse.py
```

The app will open at `http://localhost:8501`.

---

## 🧮 Model Formulation

**Decision variables:**
- `y[w] ∈ {0,1}` — 1 if warehouse `w` is opened
- `x[w,c] ∈ {0,1}` — 1 if city `c` is served by warehouse `w`

**Objective (minimise):**
```
Σ_w  FixedCost[w] × y[w]
+ Σ_w Σ_c  TransportCost/km × Distance[w,c] × Demand[c] × x[w,c]
```

**Constraints:**
- Each city assigned to exactly one DC: `Σ_w x[w,c] = 1  ∀c`
- City can only be assigned to an open DC: `x[w,c] ≤ y[w]  ∀w,c`
- At least one DC must be open: `Σ_w y[w] ≥ 1`
- Optional forced-open / forced-closed constraints (sidebar controls)

Solved with **CBC** (open-source MILP solver) via the **PuLP** interface.

---

## 📁 Project Structure

```
milp-warehouse-optimization/
├── milp_warehouse.py     # Main Streamlit app + MILP model
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Library | Role |
|---------|------|
| `PuLP` | MILP model definition & CBC solver interface |
| `Streamlit` | Interactive web dashboard |
| `pandas` | Tabular KPI displays |
| `NumPy` | Haversine distance computation |
| `matplotlib` | Network map & charts |

---

## 💡 Key Insights from the Model

1. **Fixed-cost dominance** — the choice of *which* warehouses to open typically accounts for 55–70% of total network spend, making location selection the highest-leverage decision.
2. **Naive (all-open) strategy is suboptimal** — opening all 6 DCs wastes crores in excess fixed costs despite minimising transport distances.
3. **Network topology is robust** — the optimal set of open warehouses is stable under ±30% demand shocks; only total spend changes.
4. **Fuel sensitivity** — a 20% fuel price increase raises total cost by ~8–12%, highlighting the value of 3PL fuel hedging clauses.

---

## 👤 Author

**Pallapolu Bhuvan Chandra**  
B.E. — BITS Pilani  
Supply Chain Management Portfolio

---

*Built as Project 3 of the SCM Analytics Portfolio.*
