"""
Complete Demonstration of Volatility Surface Modeling

This script demonstrates the full capabilities of the volatility surface library:
1. Building volatility surfaces with sticky strike/delta
2. Dupire local volatility model
3. Heston stochastic volatility model
4. SABR model
5. Arbitrage-free constraints
6. Interpolation and smoothing
7. Comprehensive visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.volatility_surface import VolatilitySurface, create_sample_surface
from src.dupire import DupireLocalVolatility
from src.heston import HestonModel, HestonParameters
from src.sabr import SABRModel, SABRSurface
from src.arbitrage import ArbitrageChecker
from src.interpolation import SurfaceInterpolator
from src.visualization import VolatilitySurfaceVisualizer


def generate_market_data(spot: float = 100.0) -> tuple:
    """
    Generate synthetic market implied volatility data.

    Creates a realistic volatility surface with:
    - Negative skew (puts more expensive)
    - Term structure effect (vol tends to increase with maturity)
    - Smile effect (OTM options have higher vol)
    """
    strikes = np.linspace(70, 130, 13)
    maturities = np.array([0.083, 0.25, 0.5, 1.0, 2.0])  # 1M, 3M, 6M, 1Y, 2Y

    implied_vols = np.zeros((len(strikes), len(maturities)))

    atm_vol = 0.20
    skew = -0.002
    smile = 0.0008
    term_slope = 0.02

    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            moneyness = (K - spot) / spot
            vol = atm_vol
            vol += skew * moneyness
            vol += smile * moneyness**2
            vol += term_slope * np.sqrt(T)
            vol += np.random.normal(0, 0.002)
            implied_vols[i, j] = max(0.05, vol)

    return strikes, maturities, implied_vols


def demo_volatility_surface():
    """Demonstrate basic volatility surface functionality."""
    print("\n" + "="*60)
    print("VOLATILITY SURFACE DEMONSTRATION")
    print("="*60)

    spot = 100.0
    rate = 0.05

    surface = create_sample_surface(spot, rate)

    print(f"\nSpot Price: {spot}")
    print(f"Risk-free Rate: {rate}")
    print(f"Strikes: {surface.strikes[0]:.1f} to {surface.strikes[-1]:.1f}")
    print(f"Maturities: {surface.maturities}")

    K_test, T_test = 95.0, 0.5
    vol = surface.get_implied_vol(K_test, T_test)
    print(f"\nInterpolated IV at K={K_test}, T={T_test}: {vol*100:.2f}%")

    atm_vol = surface.get_atm_vol(T_test)
    print(f"ATM IV at T={T_test}: {atm_vol*100:.2f}%")

    return surface


def demo_dupire_model():
    """Demonstrate Dupire local volatility model."""
    print("\n" + "="*60)
    print("DUPIRE LOCAL VOLATILITY MODEL")
    print("="*60)

    print("""
Mathematical Framework:
-----------------------
The Dupire model derives local volatility as a function of spot and time:

    dS_t = r·S_t·dt + σ_loc(S_t, t)·S_t·dW_t

The local volatility is derived from call prices using the Dupire PDE:

    σ²_Dupire(K,T) = [∂C/∂T + (r-q)K·∂C/∂K + q·C] / [0.5·K²·∂²C/∂K²]

Key insight: Local vol is parameterized by SPOT price and time, not strike.
""")

    spot = 100.0
    rate = 0.05

    strikes, maturities, implied_vols = generate_market_data(spot)

    dupire = DupireLocalVolatility(spot, rate)
    dupire.calibrate_from_implied_vols(strikes, maturities, implied_vols)

    print("Calibrated local volatility surface:")
    print(f"  Shape: {dupire.local_vols.shape}")
    print(f"  Min local vol: {dupire.local_vols.min()*100:.2f}%")
    print(f"  Max local vol: {dupire.local_vols.max()*100:.2f}%")

    S_test, t_test = 100.0, 0.5
    local_vol = dupire.get_local_volatility(S_test, t_test)
    print(f"\nLocal vol at S={S_test}, t={t_test}: {local_vol*100:.2f}%")

    return dupire


def demo_heston_model():
    """Demonstrate Heston stochastic volatility model."""
    print("\n" + "="*60)
    print("HESTON STOCHASTIC VOLATILITY MODEL")
    print("="*60)

    print("""
Mathematical Framework:
-----------------------
The Heston model describes correlated dynamics for price and variance:

    dS_t = r·S_t·dt + √v_t·S_t·dW^S_t
    dv_t = κ(θ - v_t)dt + σ·√v_t·dW^v_t

CRITICAL: The correlation between Wiener processes:

    dW^S_t · dW^v_t = ρ·dt

This correlation controls the skew of the implied volatility smile:
- Negative ρ → Negative skew (leverage effect)
- Positive ρ → Positive skew

Feller Condition: 2κθ > σ² ensures variance stays positive.
""")

    spot = 100.0
    rate = 0.05

    heston = HestonModel(spot, rate)

    v0 = 0.04        # Initial variance (20% vol)
    kappa = 2.0      # Mean reversion speed
    theta = 0.04     # Long-term variance
    sigma = 0.3      # Vol of vol
    rho = -0.7       # Negative correlation (leverage effect)

    heston.set_parameters(v0, kappa, theta, sigma, rho)

    print(f"\nHeston Parameters:")
    print(f"  v0 (initial variance): {v0} → σ0 = {np.sqrt(v0)*100:.1f}%")
    print(f"  κ (mean reversion):    {kappa}")
    print(f"  θ (long-term var):     {theta} → σ∞ = {np.sqrt(theta)*100:.1f}%")
    print(f"  σ (vol of vol):        {sigma}")
    print(f"  ρ (correlation):       {rho}")
    print(f"  Feller ratio:          {heston.params.feller_ratio:.2f} (should be > 1)")

    K, T = 95.0, 0.5
    call_price = heston.price_european_call(K, T)
    implied_vol = heston.implied_volatility(K, T)

    print(f"\nOption Pricing (K={K}, T={T}):")
    print(f"  Call price:    ${call_price:.4f}")
    print(f"  Implied vol:   {implied_vol*100:.2f}%")

    strikes = np.linspace(80, 120, 9)
    smile = np.array([heston.implied_volatility(K, T) for K in strikes])

    print(f"\nVolatility Smile at T={T}:")
    for K, vol in zip(strikes, smile):
        print(f"  K={K:6.1f}: {vol*100:5.2f}%")

    return heston


def demo_sabr_model():
    """Demonstrate SABR stochastic volatility model."""
    print("\n" + "="*60)
    print("SABR MODEL")
    print("="*60)

    print("""
Mathematical Framework:
-----------------------
The SABR model operates in FORWARD price space (not spot):

    dF_t = σ_t · F_t^β · dW_t    (NO DRIFT - forward is martingale)
    dσ_t = ν · σ_t · dZ_t

Correlation: dW_t · dZ_t = ρ·dt

The implied volatility approximation (Hagan et al., 2002):

    σ_SABR(K,T) = (α / (FK)^((1-β)/2)) · [1 + corrections] · (z/x(z))

Parameters:
- α (alpha): Initial volatility level
- β (beta):  CEV exponent (0=normal, 1=lognormal)
- ρ (rho):   Correlation → controls skew
- ν (nu):    Vol of vol → controls smile curvature
""")

    forward = 100.0
    maturity = 1.0
    rate = 0.05

    sabr = SABRModel(forward, maturity, rate)

    alpha = 0.3      # Initial vol
    beta = 0.5       # CEV exponent
    rho = -0.3       # Correlation
    nu = 0.4         # Vol of vol

    sabr.set_parameters(alpha, beta, rho, nu)

    print(f"\nSABR Parameters:")
    print(f"  α (alpha): {alpha}")
    print(f"  β (beta):  {beta}")
    print(f"  ρ (rho):   {rho}")
    print(f"  ν (nu):    {nu}")

    print(f"\nATM Implied Vol: {sabr.implied_volatility_atm()*100:.2f}%")

    strikes = np.linspace(80, 120, 9)
    smile = np.array([sabr.implied_volatility(K) for K in strikes])

    print(f"\nVolatility Smile at T={maturity}:")
    for K, vol in zip(strikes, smile):
        print(f"  K={K:6.1f}: {vol*100:5.2f}%")

    return sabr


def demo_arbitrage_constraints():
    """Demonstrate arbitrage-free constraints."""
    print("\n" + "="*60)
    print("ARBITRAGE-FREE CONSTRAINTS")
    print("="*60)

    print("""
No-Arbitrage Conditions:
------------------------

1. CALENDAR SPREAD ARBITRAGE
   Total variance must increase with maturity:
   ∂(σ²T)/∂T ≥ 0

2. BUTTERFLY SPREAD ARBITRAGE
   Call price must be convex in strike:
   ∂²C/∂K² ≥ 0

3. CALL SPREAD ARBITRAGE
   Call prices must decrease with strike:
   ∂C/∂K ≤ 0
""")

    spot = 100.0
    rate = 0.05

    strikes, maturities, implied_vols = generate_market_data(spot)

    implied_vols_bad = implied_vols.copy()
    implied_vols_bad[:, 2] = implied_vols_bad[:, 3] * 1.1

    checker = ArbitrageChecker(spot, rate)

    print("\nChecking original surface for arbitrage...")
    violations_orig = checker.check_all(strikes, maturities, implied_vols)
    print(f"  Violations found: {len(violations_orig)}")

    print("\nChecking modified surface (with calendar violations)...")
    violations_bad = checker.check_all(strikes, maturities, implied_vols_bad)
    print(f"  Violations found: {len(violations_bad)}")

    if violations_bad:
        print("\n  Sample violations:")
        for v in violations_bad[:3]:
            print(f"    - {v.violation_type}: {v.description[:60]}...")

    print("\nEnforcing arbitrage-free constraints...")
    corrected_vols = checker.enforce_all_constraints(strikes, maturities, implied_vols_bad)
    violations_corrected = checker.check_all(strikes, maturities, corrected_vols)
    print(f"  Violations after correction: {len(violations_corrected)}")

    return checker


def demo_interpolation():
    """Demonstrate interpolation and smoothing techniques."""
    print("\n" + "="*60)
    print("INTERPOLATION AND SMOOTHING")
    print("="*60)

    print("""
Interpolation Methods:
----------------------
1. Cubic Splines: C² continuity, smooth
2. Total Variance: Interpolates w=σ²T (preserves calendar arb)
3. Gaussian Kernel: Non-parametric smoothing

Extrapolation Methods:
----------------------
1. Flat: Extend boundary values
2. Log-linear: Linear in log-strike space
3. Asymptotic: Uses Lee's moment formula
""")

    spot = 100.0
    rate = 0.05

    strikes, maturities, implied_vols = generate_market_data(spot)

    interpolator = SurfaceInterpolator(
        strikes, maturities, implied_vols, spot, rate
    )

    interpolator.build_interpolator(method='total_variance')

    K_test, T_test = 95.0, 0.75
    vol_interp = interpolator.interpolate(K_test, T_test)
    print(f"\nInterpolated vol at K={K_test}, T={T_test}: {vol_interp*100:.2f}%")

    K_extrap = 60.0
    vol_flat = interpolator.interpolate(K_extrap, T_test, extrapolation='flat')
    vol_asymp = interpolator.interpolate(K_extrap, T_test, extrapolation='asymptotic')
    print(f"\nExtrapolated vol at K={K_extrap}, T={T_test}:")
    print(f"  Flat:       {vol_flat*100:.2f}%")
    print(f"  Asymptotic: {vol_asymp*100:.2f}%")

    print("\nApplying Gaussian smoothing...")
    smoothed_vols = interpolator.smooth_surface(method='gaussian', sigma_strike=1.0, sigma_time=0.5)

    print("\nFitting SVI parameterization...")
    svi_params = interpolator.fit_svi(maturities[2])
    print(f"  SVI parameters at T={maturities[2]}:")
    for key, val in svi_params.items():
        if key != 'maturity':
            print(f"    {key}: {val:.4f}")

    return interpolator


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)

    spot = 100.0
    rate = 0.05

    strikes, maturities, implied_vols = generate_market_data(spot)

    viz = VolatilitySurfaceVisualizer(figsize=(12, 8))

    print("\nCreating visualizations...")

    fig1 = viz.plot_surface_3d(
        strikes, maturities, implied_vols,
        title="Implied Volatility Surface",
        spot=spot
    )
    fig1.savefig('vol_surface_3d.png', dpi=150, bbox_inches='tight')
    print("  Saved: vol_surface_3d.png")

    fig2 = viz.plot_contour(
        strikes, maturities, implied_vols,
        title="Volatility Surface Contour"
    )
    fig2.savefig('vol_surface_contour.png', dpi=150, bbox_inches='tight')
    print("  Saved: vol_surface_contour.png")

    fig3 = viz.plot_multiple_smiles(
        strikes, maturities, implied_vols,
        title="Volatility Smiles Across Maturities"
    )
    fig3.savefig('vol_smiles.png', dpi=150, bbox_inches='tight')
    print("  Saved: vol_smiles.png")

    atm_idx = np.argmin(np.abs(strikes - spot))
    strike_indices = [0, atm_idx, -1]
    fig4 = viz.plot_term_structure(
        maturities,
        implied_vols[strike_indices, :],
        strikes=strikes[strike_indices],
        strike_labels=['Low Strike', 'ATM', 'High Strike'],
        title="Volatility Term Structure"
    )
    fig4.savefig('vol_term_structure.png', dpi=150, bbox_inches='tight')
    print("  Saved: vol_term_structure.png")

    fig5 = viz.create_dashboard(
        strikes, maturities, implied_vols, spot,
        title="Volatility Surface Dashboard"
    )
    fig5.savefig('vol_dashboard.png', dpi=150, bbox_inches='tight')
    print("  Saved: vol_dashboard.png")

    plt.close('all')

    return viz


def demo_model_comparison():
    """Compare different volatility models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    spot = 100.0
    forward = 100.0
    rate = 0.05
    maturity = 1.0

    strikes = np.linspace(80, 120, 21)

    heston = HestonModel(spot, rate)
    heston.set_parameters(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    heston_vols = np.array([heston.implied_volatility(K, maturity) for K in strikes])

    sabr = SABRModel(forward, maturity, rate)
    sabr.set_parameters(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
    sabr_vols = np.array([sabr.implied_volatility(K) for K in strikes])

    print("\nModel comparison at T=1.0:")
    print(f"{'Strike':>8} {'Heston':>10} {'SABR':>10} {'Diff':>10}")
    print("-" * 40)
    for i in range(0, len(strikes), 4):
        K = strikes[i]
        diff = (heston_vols[i] - sabr_vols[i]) * 100
        print(f"{K:8.1f} {heston_vols[i]*100:9.2f}% {sabr_vols[i]*100:9.2f}% {diff:9.2f}%")

    viz = VolatilitySurfaceVisualizer()
    fig = viz.plot_model_comparison(
        strikes, maturity,
        heston_vols,
        {'Heston': heston_vols, 'SABR': sabr_vols},
        title="Model Comparison"
    )
    fig.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: model_comparison.png")

    plt.close('all')


def main():
    """Run all demonstrations."""
    print("\n" + "#"*60)
    print("#" + " "*20 + "VOLATILITY SURFACE" + " "*20 + "#")
    print("#" + " "*15 + "MODELING DEMONSTRATION" + " "*16 + "#")
    print("#"*60)

    surface = demo_volatility_surface()

    dupire = demo_dupire_model()

    heston = demo_heston_model()

    sabr = demo_sabr_model()

    checker = demo_arbitrage_constraints()

    interpolator = demo_interpolation()

    viz = demo_visualization()

    demo_model_comparison()

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nGenerated visualization files:")
    print("  - vol_surface_3d.png")
    print("  - vol_surface_contour.png")
    print("  - vol_smiles.png")
    print("  - vol_term_structure.png")
    print("  - vol_dashboard.png")
    print("  - model_comparison.png")


if __name__ == "__main__":
    main()
