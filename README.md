# Volatility Surface Modeling Library

A comprehensive Python library for building, analyzing, and visualizing implied volatility surfaces using industry-standard models and techniques.

## Table of Contents

- [Overview](#overview)
- [Mathematical Background](#mathematical-background)
  - [Volatility Surface](#volatility-surface)
  - [Dupire Local Volatility Model](#dupire-local-volatility-model)
  - [Heston Stochastic Volatility Model](#heston-stochastic-volatility-model)
  - [SABR Model](#sabr-model)
  - [Arbitrage-Free Constraints](#arbitrage-free-constraints)
  - [Sticky Strike vs Sticky Delta](#sticky-strike-vs-sticky-delta)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Visualization Examples](#visualization-examples)
- [API Reference](#api-reference)

---

## Overview

A volatility surface provides a **three-dimensional representation** of implied volatilities across different strikes (K) and maturities (T). This library implements:

- **Dupire Local Volatility Model** - Derives local volatility from option prices
- **Heston Stochastic Volatility Model** - Correlated price and variance dynamics
- **SABR Model** - Forward price based stochastic volatility
- **Arbitrage-Free Constraints** - Calendar and butterfly spread checks
- **Interpolation & Smoothing** - Cubic splines, Gaussian kernels, SVI fitting
- **Comprehensive Visualizations** - 3D surfaces, contours, smiles, dashboards

---

## Mathematical Background

### Volatility Surface

An implied volatility surface σ(K, T) maps the Black-Scholes implied volatility for each combination of strike K and maturity T. The surface encodes market expectations about future volatility and must satisfy no-arbitrage constraints.

```
                    │
        Implied     │     ╱╲
        Volatility  │    ╱  ╲    Volatility Smile
           (σ)      │   ╱    ╲
                    │  ╱      ╲
                    │ ╱        ╲
                    │╱──────────╲───────────
                            Strike (K)
```

---

### Dupire Local Volatility Model

In the Dupire model, we derive **local volatility as a function of spot price and time** (not strike). The underlying asset follows:

```
dSₜ = r·Sₜ·dt + σ_loc(Sₜ, t)·Sₜ·dWₜ
```

The **Dupire PDE** gives local volatility from call prices:

```
         ∂C/∂T + (r-q)K·∂C/∂K + q·C
σ²_Dupire(K,T) = ────────────────────────────
                    0.5·K²·∂²C/∂K²
```

**Key Insight**: The local volatility σ_loc(S, t) tells us the instantaneous volatility when the spot price is at level S at time t. Strike K in the Dupire formula corresponds to the spot level in the local volatility function.

**Convexity Constraint**: For the denominator to be positive (no-arbitrage):
```
∂²C/∂K² ≥ 0
```

---

### Heston Stochastic Volatility Model

The Heston model introduces **stochastic variance** with mean reversion:

```
dSₜ = r·Sₜ·dt + √vₜ·Sₜ·dWₜˢ

dvₜ = κ(θ - vₜ)dt + σ·√vₜ·dWₜᵛ
```

**CRITICAL**: The correlation between the two Wiener processes:

```
dWₜˢ · dWₜᵛ = ρ·dt
```

This correlation term **significantly influences the skew and curvature** of the implied volatility surface:

| Parameter | Effect on Surface |
|-----------|-------------------|
| ρ < 0 | Negative skew (leverage effect) |
| ρ > 0 | Positive skew |
| σ (vol of vol) | Controls smile curvature |
| κ (mean reversion) | Controls term structure |
| θ (long-term var) | Sets long-term vol level |

**Feller Condition**: For variance to remain strictly positive:
```
2κθ > σ²
```

---

### SABR Model

The SABR model operates in **forward price space** (not spot price). Since the forward is a martingale under the forward measure, there is **no drift term**:

```
dFₜ = σₜ·Fₜᵝ·dWₜ     (NO DRIFT)

dσₜ = ν·σₜ·dZₜ
```

**Implied Volatility Approximation** (Hagan et al., 2002):

```
                α                        (1-β)² log²(F/K)      ν²
σ_SABR(K,T) = ──────────── · [1 + ────────────────── + ──────────────── + ...] · z/x(z)
              (FK)^((1-β)/2)           24·(FK)^(1-β)    24α²(FK)^(1-β)
```

where:
```
z = (ν/α)·(FK)^((1-β)/2)·log(F/K)

x(z) = log[(√(1-2ρz+z²) + z - ρ) / (1-ρ)]
```

**For ATM options** (F = K):
```
σ_ATM = α/F^(1-β) · [1 + ((1-β)²α²/(24F^(2-2β)) + ρβνα/(4F^(1-β)) + (2-3ρ²)ν²/24)·T]
```

---

### Arbitrage-Free Constraints

#### Calendar Spread Arbitrage

Implied variance must **increase with maturity**:

```
∂σ(K,T)/∂T ≥ -σ/(2T)
```

Or equivalently, total variance w = σ²T must be non-decreasing:
```
w(K, T₂) ≥ w(K, T₁)  for T₂ > T₁
```

#### Butterfly Spread Arbitrage

The second derivative of call price with respect to strike must be **non-negative**:

```
∂²C(S,T)/∂K² ≥ 0
```

This ensures the risk-neutral probability density is non-negative.

---

### Sticky Strike vs Sticky Delta

**Sticky Strike**: The volatility surface is parameterized based on **absolute strike levels**. Implied volatilities remain fixed for specific strikes even as the spot price changes.

```
σ(K, T) is fixed for each K
```

**Sticky Delta**: The volatility surface is parameterized based on **delta levels** (relative to spot).

```
σ(Δ, T) where Δ = ∂C/∂S
```

For a call option:
```
Δ_call = e^(-qT) · N(d₁)

where d₁ = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
```

This approach is more dynamic, adjusting volatilities based on the relative position of strike to spot.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Volatility-Surface-.git
cd Volatility-Surface-

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- numpy >= 1.21.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- plotly >= 5.0.0 (optional, for interactive plots)
- seaborn >= 0.11.0
- numba >= 0.54.0

---

## Quick Start

```python
from src.volatility_surface import VolatilitySurface, create_sample_surface
from src.dupire import DupireLocalVolatility
from src.heston import HestonModel
from src.sabr import SABRModel
from src.visualization import VolatilitySurfaceVisualizer

# Create a sample volatility surface
spot = 100.0
surface = create_sample_surface(spot=spot, base_vol=0.20, skew=-0.1)

# Get implied volatility at specific strike and maturity
vol = surface.get_implied_vol(strike=95.0, maturity=0.5)
print(f"Implied Vol: {vol*100:.2f}%")

# Visualize the surface
viz = VolatilitySurfaceVisualizer()
fig = viz.plot_surface_3d(
    surface.strikes,
    surface.maturities,
    surface.implied_vols,
    spot=spot
)
fig.savefig('volatility_surface.png')
```

---

## Detailed Usage

### Dupire Local Volatility

```python
from src.dupire import DupireLocalVolatility
import numpy as np

# Market data
spot = 100.0
strikes = np.linspace(70, 130, 15)
maturities = np.array([0.25, 0.5, 1.0, 2.0])
implied_vols = ...  # Your market implied vols (shape: strikes x maturities)

# Calibrate Dupire model
dupire = DupireLocalVolatility(spot, rate=0.05)
dupire.calibrate_from_implied_vols(strikes, maturities, implied_vols)

# Get local volatility at spot=100, time=0.5
local_vol = dupire.get_local_volatility(100.0, 0.5)
print(f"Local Vol: {local_vol*100:.2f}%")

# Price option using Monte Carlo with local vol
price = dupire.price_european_option(strike=100, maturity=1.0, option_type='call')
```

### Heston Model

```python
from src.heston import HestonModel

# Initialize model
heston = HestonModel(spot=100.0, rate=0.05)

# Set parameters
heston.set_parameters(
    v0=0.04,      # Initial variance (20% vol)
    kappa=2.0,    # Mean reversion speed
    theta=0.04,   # Long-term variance
    sigma=0.3,    # Vol of vol
    rho=-0.7      # Correlation (negative = leverage effect)
)

# Price options using semi-closed form
call_price = heston.price_european_call(strike=100, maturity=1.0)

# Get implied volatility
implied_vol = heston.implied_volatility(strike=95, maturity=0.5)

# Generate volatility surface
strikes = np.linspace(80, 120, 11)
maturities = np.array([0.25, 0.5, 1.0])
vol_surface = heston.generate_implied_vol_surface(strikes, maturities)

# Simulate paths
S_paths, v_paths = heston.simulate_paths(maturity=1.0, n_paths=10000)
```

### SABR Model

```python
from src.sabr import SABRModel, SABRSurface

# Initialize SABR (operates in forward space)
forward = 100.0
maturity = 1.0
sabr = SABRModel(forward, maturity, rate=0.05)

# Set parameters
sabr.set_parameters(
    alpha=0.2,    # Initial vol
    beta=0.5,     # CEV exponent (0=normal, 1=lognormal)
    rho=-0.3,     # Correlation
    nu=0.4        # Vol of vol
)

# Get implied volatility
vol = sabr.implied_volatility(strike=95)

# Generate smile
strikes, vols = sabr.generate_smile(n_strikes=50)

# Calibrate to market data
market_strikes = np.array([90, 95, 100, 105, 110])
market_vols = np.array([0.22, 0.20, 0.19, 0.20, 0.21])
sabr.calibrate(market_strikes, market_vols, fix_beta=0.5)
```

### Arbitrage Checking

```python
from src.arbitrage import ArbitrageChecker

checker = ArbitrageChecker(spot=100.0, rate=0.05)

# Check all arbitrage conditions
violations = checker.check_all(strikes, maturities, implied_vols)

if violations:
    print(f"Found {len(violations)} violations:")
    for v in violations:
        print(f"  {v.violation_type}: {v.description}")

# Enforce arbitrage-free constraints
corrected_vols = checker.enforce_all_constraints(strikes, maturities, implied_vols)
```

### Interpolation and Smoothing

```python
from src.interpolation import SurfaceInterpolator

interpolator = SurfaceInterpolator(
    strikes, maturities, implied_vols,
    spot=100.0, rate=0.05
)

# Build interpolator using total variance (preserves calendar arbitrage)
interpolator.build_interpolator(method='total_variance')

# Interpolate at any point
vol = interpolator.interpolate(strike=95.5, maturity=0.75)

# Extrapolate for extreme strikes
vol_wing = interpolator.interpolate(strike=60, maturity=1.0, extrapolation='asymptotic')

# Apply smoothing
smoothed_surface = interpolator.smooth_surface(method='gaussian', sigma_strike=1.0)

# Fit SVI parameterization
svi_params = interpolator.fit_svi(maturity=1.0)
```

---

## Visualization Examples

### 3D Surface Plot

```python
from src.visualization import VolatilitySurfaceVisualizer

viz = VolatilitySurfaceVisualizer()

# Static matplotlib plot
fig = viz.plot_surface_3d(
    strikes, maturities, implied_vols,
    title="Implied Volatility Surface",
    spot=100.0
)

# Interactive plotly plot
fig_interactive = viz.plot_surface_3d(
    strikes, maturities, implied_vols,
    interactive=True
)
```

### Volatility Smile

```python
# Single smile
fig = viz.plot_smile(
    strikes, implied_vols[:, 2],  # Select maturity index
    maturity=0.5,
    spot=100.0,
    show_moneyness=True
)

# Multiple smiles comparison
fig = viz.plot_multiple_smiles(strikes, maturities, implied_vols)
```

### Model Comparison

```python
# Compare Heston vs SABR
model_vols = {
    'Heston': heston_vols,
    'SABR': sabr_vols
}
fig = viz.plot_model_comparison(strikes, maturity=1.0, market_vols, model_vols)
```

### Comprehensive Dashboard

```python
fig = viz.create_dashboard(
    strikes, maturities, implied_vols,
    spot=100.0,
    title="Volatility Surface Dashboard"
)
```

---

## API Reference

### `VolatilitySurface`

| Method | Description |
|--------|-------------|
| `set_data(strikes, maturities, implied_vols)` | Set surface data |
| `get_implied_vol(strike, maturity)` | Interpolate implied vol |
| `get_atm_vol(maturity)` | Get ATM volatility |
| `get_moneyness(strike, maturity)` | Calculate log-moneyness |
| `total_variance(strike, maturity)` | Calculate σ²T |
| `shift_spot(new_spot)` | Update spot price |

### `DupireLocalVolatility`

| Method | Description |
|--------|-------------|
| `calibrate_from_implied_vols(...)` | Calibrate from market data |
| `get_local_volatility(spot, time)` | Get local vol at (S, t) |
| `price_european_option(...)` | Price via Monte Carlo |

### `HestonModel`

| Method | Description |
|--------|-------------|
| `set_parameters(v0, kappa, theta, sigma, rho)` | Set model parameters |
| `price_european_call(strike, maturity)` | Semi-closed form pricing |
| `implied_volatility(strike, maturity)` | Extract implied vol |
| `simulate_paths(maturity, n_paths)` | Monte Carlo simulation |
| `calibrate(strikes, maturities, prices)` | Calibrate to market |

### `SABRModel`

| Method | Description |
|--------|-------------|
| `set_parameters(alpha, beta, rho, nu)` | Set model parameters |
| `implied_volatility(strike)` | Hagan's approximation |
| `generate_smile(n_strikes)` | Generate vol smile |
| `calibrate(strikes, market_vols)` | Calibrate to market |

---

## Project Structure

```
Volatility-Surface-/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── volatility_surface.py   # Base surface class
│   ├── dupire.py               # Dupire local vol model
│   ├── heston.py               # Heston stochastic vol
│   ├── sabr.py                 # SABR model
│   ├── arbitrage.py            # Arbitrage constraints
│   ├── interpolation.py        # Interpolation methods
│   └── visualization.py        # Plotting utilities
└── examples/
    ├── __init__.py
    └── complete_demo.py        # Full demonstration
```

---

## References

1. Dupire, B. (1994). "Pricing with a Smile". Risk Magazine.
2. Heston, S.L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility".
3. Hagan, P.S. et al. (2002). "Managing Smile Risk". Wilmott Magazine.
4. Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide".
5. Lee, R.W. (2004). "The Moment Formula for Implied Volatility at Extreme Strikes".

---

## License

MIT License

---

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
