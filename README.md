
---

# **Hull‑White Interactive Rate Simulation**
An application for simulating short‑rate dynamics under the **Hull–White one‑factor model**, visualizing scenario paths and distributions, and interactively reshaping the resulting distribution to generate a new, model‑consistent scenario set.

---

## **Overview**
This project provides an end‑to‑end workflow for interest‑rate scenario generation using the Hull–White model. Starting from a user‑provided interest‑rate curve (e.g., 1M, 3M, 6M, 1Y, …), the application:

1. **Bootstraps the initial curve** and calibrates the Hull–White model parameters.  
2. **Simulates short‑rate paths** and corresponding yield curves.  
3. **Displays an interactive dashboard** showing:
   - A selection of simulated short‑rate paths  
   - The resulting distribution at a chosen horizon  
   - Yield curve evolution  
4. **Allows the user to reshape the distribution manually** by dragging points or adjusting the density curve.  
5. **Fits a functional form** (e.g., shifted lognormal, spline‑based density, mixture model) to the user‑modified distribution.  
6. **Re‑simulates scenarios** consistent with the new distribution, producing a final scenario set suitable for discounting, valuation, or risk analysis.

---

## **Key Features**
### **Hull–White Simulation**
- One‑factor Hull–White short‑rate model  
  \[
  dr_t = \theta(t)\,dt + a(b - r_t)\,dt + \sigma\,dW_t
  \]
- Calibration to the initial curve via analytical bond‑pricing relationships  
- Generation of:
  - Short‑rate paths  
  - Zero‑coupon yield curves  
  - Forward‑rate curves  

### **Interactive Dashboard**
- Visualization of:
  - A subset of simulated paths  
  - Distribution of short rates at a chosen horizon  
  - Yield curve snapshots across time  
- User‑driven distribution editing:
  - Drag points on the density curve  
  - Adjust quantiles  
  - Modify tail behavior  
  - Smooth or reshape the distribution interactively  

### **Distribution Fitting Engine**
- Takes the user‑modified distribution and finds the **closest functional form** using:
  - Parametric families (lognormal, shifted lognormal, generalized beta, etc.)  
  - Non‑parametric smoothing splines  
  - Mixture models  
- Ensures the resulting distribution is:
  - Valid (positive density, integrates to 1)  
  - Smooth  
  - Suitable for scenario generation  

### **Scenario Regeneration**
- Re‑simulates short‑rate paths consistent with the fitted distribution  
- Produces:
  - Updated short‑rate scenarios  
  - Updated yield curves  
  - Exportable scenario sets for valuation or risk engines  

---

## **Input Requirements**
The user provides a simple interest‑rate curve with tenor points such as:

```
1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, ...
```

The application handles:
- Curve interpolation  
- Bootstrapping  
- Hull–White calibration  
- Simulation setup  

No additional configuration is required.

---

## **Architecture**
### **Core Modules**
- `curve_bootstrap/` — curve interpolation and zero‑curve construction  
- `hw_model/` — Hull–White calibration and simulation  
- `dashboard/` — interactive visualization and distribution editing  
- `distribution_fit/` — fitting engine for user‑modified distributions  
- `scenario_engine/` — re‑simulation and scenario export  

### **Technology Stack**
- Python (NumPy, SciPy, pandas)  
- Visualization: Plotly / Bokeh  
- Dashboard: Dash / Streamlit  
- Optional: Rust or C++ backend for high‑performance simulation  

---

## **Usage**
### **1. Provide an Interest‑Rate Curve**
Upload or input a simple tenor–rate table.

### **2. Run Initial Simulation**
The application:
- Calibrates the Hull–White model  
- Generates short‑rate and yield‑curve paths  
- Displays the results  

### **3. Modify the Distribution**
Use the dashboard to reshape the distribution:
- Drag points  
- Adjust quantiles  
- Modify tails  

### **4. Fit and Re‑Simulate**
The model finds the best functional form and regenerates scenarios.

### **5. Export**
Download:
- Final short‑rate scenarios  
- Final yield‑curve scenarios  
- Fitted distribution parameters  

---

## **Roadmap**
- Multi‑factor Hull–White extension  
- Support for stochastic volatility  
- PCA‑based curve dynamics  
- Integration with valuation engines  
- Scenario stress testing and shock overlays  

---

## **Contributing**
Contributions are welcome. Please open an issue or submit a pull request with a clear description of your proposal.

