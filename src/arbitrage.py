"""
Arbitrage-Free Constraints for Volatility Surfaces

This module implements checks and corrections for ensuring volatility surfaces
satisfy no-arbitrage conditions.
"""

import numpy as np
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ArbitrageViolation:
    """Container for arbitrage violation information."""
    violation_type: str  # 'calendar', 'butterfly', or 'call_spread'
    location: Tuple[float, float]  # (strike, maturity) or (maturity1, maturity2)
    severity: float  # magnitude of violation
    description: str


class ArbitrageChecker:
    """
    Check and enforce arbitrage-free constraints on volatility surfaces.

    Mathematical Background:
    ------------------------
    A volatility surface must satisfy several no-arbitrage constraints:

    1. CALENDAR SPREAD ARBITRAGE
    ----------------------------
    Total implied variance must be non-decreasing in maturity:

        ∂(σ²T)/∂T ≥ 0  or equivalently  ∂σ/∂T ≥ -σ/(2T)

    This means you cannot profit by selling a near-term option and buying
    a longer-term option at the same strike.

    In terms of implied volatility:
        σ(K, T₂)²·T₂ ≥ σ(K, T₁)²·T₁  for T₂ > T₁

    2. BUTTERFLY SPREAD ARBITRAGE
    -----------------------------
    The second derivative of call price with respect to strike must be
    non-negative (convexity condition):

        ∂²C/∂K² ≥ 0

    This ensures the probability density function is non-negative.

    In terms of implied volatility, this translates to a complex condition
    on the volatility smile. For the volatility surface to be arbitrage-free:

        g(k) = (1 - k·w'/(2w))² - w'/4·(1/w + 1/4) + w''/2 ≥ 0

    where w = σ²T is total variance and k = log(K/F) is log-moneyness.

    3. CALL SPREAD ARBITRAGE
    ------------------------
    Call prices must be decreasing in strike:

        ∂C/∂K ≤ 0

    This is typically satisfied if implied volatility doesn't increase
    too rapidly with strike.

    4. PUT-CALL PARITY
    ------------------
    Not a direct constraint on the surface, but must be satisfied:

        C - P = S·e^(-qT) - K·e^(-rT)
    """

    def __init__(
        self,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0
    ):
        """
        Initialize the arbitrage checker.

        Args:
            spot: Current spot price
            rate: Risk-free interest rate
            dividend_yield: Continuous dividend yield
        """
        self.spot = spot
        self.rate = rate
        self.dividend_yield = dividend_yield

    def check_calendar_spread(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        tolerance: float = 1e-6
    ) -> List[ArbitrageViolation]:
        """
        Check for calendar spread arbitrage violations.

        Condition: σ(K, T₂)²·T₂ ≥ σ(K, T₁)²·T₁ for T₂ > T₁

        This ensures implied variance increases with maturity.

        Args:
            strikes: Array of strikes
            maturities: Array of maturities (must be sorted)
            implied_vols: 2D array of implied vols (strikes x maturities)
            tolerance: Tolerance for violations

        Returns:
            List of ArbitrageViolation objects
        """
        violations = []

        for i, K in enumerate(strikes):
            for j in range(len(maturities) - 1):
                T1, T2 = maturities[j], maturities[j + 1]
                sigma1, sigma2 = implied_vols[i, j], implied_vols[i, j + 1]

                var1 = sigma1**2 * T1  # Total variance at T1
                var2 = sigma2**2 * T2  # Total variance at T2

                if var2 < var1 - tolerance:
                    violations.append(ArbitrageViolation(
                        violation_type='calendar',
                        location=(K, T1),
                        severity=var1 - var2,
                        description=f"Calendar spread violation at K={K:.2f}: "
                                  f"w(T1={T1:.3f})={var1:.6f} > w(T2={T2:.3f})={var2:.6f}"
                    ))

        return violations

    def check_butterfly_spread(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        tolerance: float = 1e-8
    ) -> List[ArbitrageViolation]:
        """
        Check for butterfly spread arbitrage violations.

        Condition: ∂²C/∂K² ≥ 0

        This ensures the risk-neutral density is non-negative.

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            implied_vols: 2D array of implied vols
            tolerance: Tolerance for violations

        Returns:
            List of ArbitrageViolation objects
        """
        violations = []

        for j, T in enumerate(maturities):
            for i in range(1, len(strikes) - 1):
                K = strikes[i]
                K_down, K_up = strikes[i - 1], strikes[i + 1]
                dK = (K_up - K_down) / 2

                sigma = implied_vols[i, j]
                sigma_down = implied_vols[i - 1, j]
                sigma_up = implied_vols[i + 1, j]

                C = self._black_scholes_call(K, T, sigma)
                C_down = self._black_scholes_call(K_down, T, sigma_down)
                C_up = self._black_scholes_call(K_up, T, sigma_up)

                d2C_dK2 = (C_up - 2 * C + C_down) / (dK**2)

                if d2C_dK2 < -tolerance:
                    violations.append(ArbitrageViolation(
                        violation_type='butterfly',
                        location=(K, T),
                        severity=abs(d2C_dK2),
                        description=f"Butterfly violation at K={K:.2f}, T={T:.3f}: "
                                  f"∂²C/∂K² = {d2C_dK2:.8f} < 0"
                    ))

        return violations

    def check_call_spread(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        tolerance: float = 1e-8
    ) -> List[ArbitrageViolation]:
        """
        Check for call spread arbitrage violations.

        Condition: ∂C/∂K ≤ 0 (call prices decrease with strike)

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            implied_vols: 2D array of implied vols
            tolerance: Tolerance for violations

        Returns:
            List of ArbitrageViolation objects
        """
        violations = []

        for j, T in enumerate(maturities):
            for i in range(len(strikes) - 1):
                K1, K2 = strikes[i], strikes[i + 1]
                sigma1, sigma2 = implied_vols[i, j], implied_vols[i + 1, j]

                C1 = self._black_scholes_call(K1, T, sigma1)
                C2 = self._black_scholes_call(K2, T, sigma2)

                if C2 > C1 + tolerance:
                    violations.append(ArbitrageViolation(
                        violation_type='call_spread',
                        location=(K1, T),
                        severity=C2 - C1,
                        description=f"Call spread violation at T={T:.3f}: "
                                  f"C(K={K2:.2f})={C2:.4f} > C(K={K1:.2f})={C1:.4f}"
                    ))

        return violations

    def check_all(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray
    ) -> List[ArbitrageViolation]:
        """
        Run all arbitrage checks.

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            implied_vols: 2D array of implied vols

        Returns:
            List of all ArbitrageViolation objects
        """
        violations = []
        violations.extend(self.check_calendar_spread(strikes, maturities, implied_vols))
        violations.extend(self.check_butterfly_spread(strikes, maturities, implied_vols))
        violations.extend(self.check_call_spread(strikes, maturities, implied_vols))
        return violations

    def is_arbitrage_free(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray
    ) -> bool:
        """Check if surface is arbitrage-free."""
        return len(self.check_all(strikes, maturities, implied_vols)) == 0

    def enforce_calendar_arbitrage(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        method: str = 'project'
    ) -> np.ndarray:
        """
        Enforce calendar spread arbitrage-free condition.

        Ensures: σ(K, T₂)²·T₂ ≥ σ(K, T₁)²·T₁

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            implied_vols: 2D array of implied vols
            method: 'project' to project variance, 'smooth' for smoothing

        Returns:
            Corrected implied volatility surface
        """
        corrected_vols = implied_vols.copy()

        for i in range(len(strikes)):
            total_var = corrected_vols[i, :]**2 * maturities

            for j in range(1, len(maturities)):
                if total_var[j] < total_var[j - 1]:
                    if method == 'project':
                        total_var[j] = total_var[j - 1] + 1e-8
                    else:
                        avg_var = (total_var[j - 1] + total_var[j]) / 2
                        total_var[j - 1] = avg_var
                        total_var[j] = avg_var + 1e-8

            corrected_vols[i, :] = np.sqrt(total_var / maturities)

        return corrected_vols

    def enforce_butterfly_arbitrage(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        max_iterations: int = 100
    ) -> np.ndarray:
        """
        Enforce butterfly spread arbitrage-free condition.

        Uses iterative smoothing to ensure ∂²C/∂K² ≥ 0.

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            implied_vols: 2D array of implied vols
            max_iterations: Maximum smoothing iterations

        Returns:
            Corrected implied volatility surface
        """
        corrected_vols = implied_vols.copy()

        for _ in range(max_iterations):
            violations_found = False

            for j, T in enumerate(maturities):
                for i in range(1, len(strikes) - 1):
                    K = strikes[i]
                    dK = (strikes[i + 1] - strikes[i - 1]) / 2

                    C = self._black_scholes_call(K, T, corrected_vols[i, j])
                    C_down = self._black_scholes_call(strikes[i - 1], T, corrected_vols[i - 1, j])
                    C_up = self._black_scholes_call(strikes[i + 1], T, corrected_vols[i + 1, j])

                    d2C_dK2 = (C_up - 2 * C + C_down) / (dK**2)

                    if d2C_dK2 < 0:
                        violations_found = True
                        corrected_vols[i, j] = 0.5 * (corrected_vols[i - 1, j] + corrected_vols[i + 1, j])

            if not violations_found:
                break

        return corrected_vols

    def enforce_all_constraints(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray
    ) -> np.ndarray:
        """
        Enforce all arbitrage-free constraints.

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            implied_vols: 2D array of implied vols

        Returns:
            Arbitrage-free implied volatility surface
        """
        corrected = self.enforce_calendar_arbitrage(strikes, maturities, implied_vols)
        corrected = self.enforce_butterfly_arbitrage(strikes, maturities, corrected)
        return corrected

    def _black_scholes_call(
        self,
        strike: float,
        maturity: float,
        sigma: float
    ) -> float:
        """Calculate Black-Scholes call price."""
        S, K, r, q, T = self.spot, strike, self.rate, self.dividend_yield, maturity

        if T <= 0 or sigma <= 0:
            return max(S - K, 0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def compute_risk_neutral_density(
        self,
        strikes: np.ndarray,
        maturity: float,
        implied_vols: np.ndarray
    ) -> np.ndarray:
        """
        Compute the risk-neutral probability density from call prices.

        By Breeden-Litzenberger formula:
            q(K) = e^(rT) · ∂²C/∂K²

        Args:
            strikes: Array of strikes
            maturity: Time to maturity
            implied_vols: Implied vols for the strikes

        Returns:
            Risk-neutral density at each strike
        """
        density = np.zeros(len(strikes))
        discount = np.exp(self.rate * maturity)

        for i in range(1, len(strikes) - 1):
            dK = (strikes[i + 1] - strikes[i - 1]) / 2

            C = self._black_scholes_call(strikes[i], maturity, implied_vols[i])
            C_down = self._black_scholes_call(strikes[i - 1], maturity, implied_vols[i - 1])
            C_up = self._black_scholes_call(strikes[i + 1], maturity, implied_vols[i + 1])

            d2C_dK2 = (C_up - 2 * C + C_down) / (dK**2)
            density[i] = discount * d2C_dK2

        density[0] = density[1]
        density[-1] = density[-2]

        return np.maximum(density, 0)


def compute_local_variance_bounds(
    spot: float,
    strike: float,
    maturity: float,
    implied_vol: float,
    dK: float,
    dT: float
) -> Tuple[float, float]:
    """
    Compute bounds on local variance from no-arbitrage conditions.

    For the Dupire local variance to be well-defined:
        σ²_loc > 0

    This requires the numerator and denominator in Dupire's formula
    to have the same sign.

    Args:
        spot: Current spot price
        strike: Option strike
        maturity: Time to maturity
        implied_vol: Implied volatility
        dK: Strike spacing
        dT: Time spacing

    Returns:
        Tuple of (lower_bound, upper_bound) for local variance
    """
    w = implied_vol**2 * maturity

    dw_dT = implied_vol**2

    lower_bound = 0.0
    upper_bound = 4 * dw_dT / (1 + w / dT)

    return lower_bound, upper_bound


def variance_interpolation_bounds(
    var_short: float,
    var_long: float,
    T_short: float,
    T_long: float,
    T_mid: float
) -> Tuple[float, float]:
    """
    Compute bounds for variance interpolation to avoid calendar arbitrage.

    For calendar arbitrage-free interpolation, total variance must be
    monotonically increasing:

        w(T_short) ≤ w(T_mid) ≤ w(T_long)

    Args:
        var_short: Total variance at shorter maturity
        var_long: Total variance at longer maturity
        T_short: Shorter maturity
        T_long: Longer maturity
        T_mid: Intermediate maturity

    Returns:
        Tuple of (min_variance, max_variance) at T_mid
    """
    w_short = var_short * T_short
    w_long = var_long * T_long

    w_mid_min = w_short
    w_mid_max = w_long

    var_mid_min = w_mid_min / T_mid
    var_mid_max = w_mid_max / T_mid

    return var_mid_min, var_mid_max
