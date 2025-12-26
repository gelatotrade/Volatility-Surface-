"""
Dupire Local Volatility Model

The Dupire model derives local volatility as a function of spot price and time,
providing a unique local volatility surface consistent with observed option prices.
"""

import numpy as np
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq
from typing import Optional, Tuple
from numba import jit


class DupireLocalVolatility:
    """
    Dupire Local Volatility Model Implementation.

    Mathematical Framework:
    -----------------------
    In the Dupire model, the underlying asset follows:

        dS_t = μ S_t dt + σ_loc(S_t, t) S_t dW_t

    where σ_loc(S, t) is the local volatility function.

    The key insight is that local volatility can be derived from call prices
    using the Dupire PDE (partial differential equation):

        σ²_Dupire(K, T) = [∂C/∂T + rK·∂C/∂K + qC] / [0.5·K²·∂²C/∂K²]

    Or equivalently in terms of implied volatility:

        σ²_loc(K, T) = [∂w/∂T] / [1 - (y/w)·∂w/∂y + 0.25·(-0.25 - 1/w + y²/w²)·(∂w/∂y)² + 0.5·∂²w/∂y²]

    where:
        - w = σ_imp² · T (total implied variance)
        - y = ln(K/F) (log-moneyness)
        - F = S·e^{(r-q)T} (forward price)

    Important Note:
    ---------------
    The local volatility is a function of SPOT price and TIME, not strike.
    The Dupire formula gives local volatility at (K, T) which corresponds to
    spot S = K at time t = T in the local volatility surface.

    Convexity Constraint:
    --------------------
    For no-arbitrage, the denominator (second derivative of call price w.r.t strike)
    must be non-negative:
        ∂²C/∂K² ≥ 0
    This is the butterfly spread arbitrage constraint.
    """

    def __init__(
        self,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0
    ):
        """
        Initialize the Dupire local volatility model.

        Args:
            spot: Current spot price
            rate: Risk-free interest rate
            dividend_yield: Continuous dividend yield
        """
        self.spot = spot
        self.rate = rate
        self.dividend_yield = dividend_yield

        self._strikes: Optional[np.ndarray] = None
        self._maturities: Optional[np.ndarray] = None
        self._implied_vols: Optional[np.ndarray] = None
        self._local_vols: Optional[np.ndarray] = None
        self._call_prices: Optional[np.ndarray] = None
        self._vol_interpolator: Optional[RectBivariateSpline] = None
        self._local_vol_interpolator: Optional[RectBivariateSpline] = None

    def calibrate_from_implied_vols(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        smooth: bool = True
    ) -> None:
        """
        Calibrate local volatility surface from implied volatilities.

        The Dupire formula in terms of implied volatility:

        σ²_loc(K,T) = [∂C/∂T + (r-q)K·∂C/∂K + qC] / [0.5·K²·∂²C/∂K²]

        where C is the Black-Scholes call price.

        Using the chain rule with implied vol σ_imp:
        ∂C/∂T = C_σ·∂σ/∂T + C_T
        ∂C/∂K = C_σ·∂σ/∂K + C_K
        ∂²C/∂K² = C_σ·∂²σ/∂K² + C_σσ·(∂σ/∂K)² + 2·C_σK·∂σ/∂K + C_KK

        Args:
            strikes: Array of strike prices
            maturities: Array of maturities
            implied_vols: 2D array of implied volatilities (strikes x maturities)
            smooth: Apply smoothing to the local volatility surface
        """
        self._strikes = np.asarray(strikes)
        self._maturities = np.asarray(maturities)
        self._implied_vols = np.asarray(implied_vols)

        self._build_vol_interpolator()
        self._compute_call_prices()
        self._compute_local_volatility()

        if smooth:
            self._smooth_local_vol_surface()

        self._build_local_vol_interpolator()

    def _build_vol_interpolator(self) -> None:
        """Build smooth implied volatility interpolator."""
        self._vol_interpolator = RectBivariateSpline(
            self._strikes,
            self._maturities,
            self._implied_vols,
            kx=min(3, len(self._strikes) - 1),
            ky=min(3, len(self._maturities) - 1),
            s=0
        )

    def _compute_call_prices(self) -> None:
        """Compute Black-Scholes call prices from implied vols."""
        self._call_prices = np.zeros_like(self._implied_vols)

        for i, K in enumerate(self._strikes):
            for j, T in enumerate(self._maturities):
                sigma = self._implied_vols[i, j]
                self._call_prices[i, j] = self._black_scholes_call(K, T, sigma)

    def _black_scholes_call(
        self,
        strike: float,
        maturity: float,
        sigma: float
    ) -> float:
        """Calculate Black-Scholes call price."""
        if maturity <= 1e-10:
            return max(self.spot - strike, 0)

        S, K, r, q, T = self.spot, strike, self.rate, self.dividend_yield, maturity

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def _compute_local_volatility(self) -> None:
        """
        Compute local volatility using the Dupire formula.

        σ²_Dupire(K, T) = [∂C/∂T + (r-q)K·∂C/∂K + q·C] / [0.5·K²·∂²C/∂K²]

        Using finite differences for numerical derivatives.
        """
        self._local_vols = np.zeros_like(self._implied_vols)

        dK = np.mean(np.diff(self._strikes))
        dT = np.mean(np.diff(self._maturities))

        for i, K in enumerate(self._strikes):
            for j, T in enumerate(self._maturities):
                sigma_imp = self._implied_vols[i, j]
                C = self._call_prices[i, j]

                dC_dT = self._compute_dC_dT(K, T, dT)
                dC_dK = self._compute_dC_dK(K, T, i, dK)
                d2C_dK2 = self._compute_d2C_dK2(K, T, i, dK)

                numerator = dC_dT + (self.rate - self.dividend_yield) * K * dC_dK + self.dividend_yield * C
                denominator = 0.5 * K**2 * d2C_dK2

                if denominator > 1e-10:
                    local_var = numerator / denominator
                    if local_var > 0:
                        self._local_vols[i, j] = np.sqrt(local_var)
                    else:
                        self._local_vols[i, j] = sigma_imp
                else:
                    self._local_vols[i, j] = sigma_imp

    def _compute_dC_dT(self, K: float, T: float, dT: float) -> float:
        """Compute ∂C/∂T using central differences."""
        T_up = T + dT
        T_down = max(T - dT, 1e-6)

        sigma_up = float(self._vol_interpolator(K, T_up)[0, 0])
        sigma_down = float(self._vol_interpolator(K, T_down)[0, 0])

        C_up = self._black_scholes_call(K, T_up, sigma_up)
        C_down = self._black_scholes_call(K, T_down, sigma_down)

        return (C_up - C_down) / (T_up - T_down)

    def _compute_dC_dK(self, K: float, T: float, i: int, dK: float) -> float:
        """Compute ∂C/∂K using central differences."""
        K_up = K + dK
        K_down = max(K - dK, self.spot * 0.01)

        sigma_up = float(self._vol_interpolator(K_up, T)[0, 0])
        sigma_down = float(self._vol_interpolator(K_down, T)[0, 0])

        C_up = self._black_scholes_call(K_up, T, sigma_up)
        C_down = self._black_scholes_call(K_down, T, sigma_down)

        return (C_up - C_down) / (K_up - K_down)

    def _compute_d2C_dK2(self, K: float, T: float, i: int, dK: float) -> float:
        """
        Compute ∂²C/∂K² using central differences.

        This is the key convexity term - must be ≥ 0 for no-arbitrage.
        """
        K_up = K + dK
        K_down = max(K - dK, self.spot * 0.01)

        sigma = float(self._vol_interpolator(K, T)[0, 0])
        sigma_up = float(self._vol_interpolator(K_up, T)[0, 0])
        sigma_down = float(self._vol_interpolator(K_down, T)[0, 0])

        C = self._black_scholes_call(K, T, sigma)
        C_up = self._black_scholes_call(K_up, T, sigma_up)
        C_down = self._black_scholes_call(K_down, T, sigma_down)

        return (C_up - 2 * C + C_down) / (dK**2)

    def _smooth_local_vol_surface(self, iterations: int = 3) -> None:
        """Apply Gaussian smoothing to local volatility surface."""
        from scipy.ndimage import gaussian_filter

        self._local_vols = gaussian_filter(self._local_vols, sigma=0.5)

        min_vol = 0.01
        max_vol = 2.0
        self._local_vols = np.clip(self._local_vols, min_vol, max_vol)

    def _build_local_vol_interpolator(self) -> None:
        """Build interpolator for local volatility surface."""
        self._local_vol_interpolator = RectBivariateSpline(
            self._strikes,
            self._maturities,
            self._local_vols,
            kx=min(3, len(self._strikes) - 1),
            ky=min(3, len(self._maturities) - 1)
        )

    def get_local_volatility(self, spot: float, time: float) -> float:
        """
        Get local volatility at a given spot price and time.

        Note: In Dupire, local vol is parameterized by spot and time,
        not strike. The local vol σ_loc(S, t) tells us the instantaneous
        volatility when the spot is at S at time t.

        Args:
            spot: Spot price level
            time: Time point

        Returns:
            Local volatility at (spot, time)
        """
        if self._local_vol_interpolator is None:
            raise ValueError("Model not calibrated. Call calibrate_from_implied_vols() first.")

        return float(self._local_vol_interpolator(spot, time)[0, 0])

    def get_local_vol_surface(
        self,
        spots: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the full local volatility surface.

        Args:
            spots: Array of spot prices (defaults to calibration strikes)
            times: Array of times (defaults to calibration maturities)

        Returns:
            Tuple of (spots, times, local_vols)
        """
        if spots is None:
            spots = self._strikes
        if times is None:
            times = self._maturities

        local_vols = np.zeros((len(spots), len(times)))

        for i, S in enumerate(spots):
            for j, t in enumerate(times):
                local_vols[i, j] = self.get_local_volatility(S, t)

        return spots, times, local_vols

    def price_european_option(
        self,
        strike: float,
        maturity: float,
        option_type: str = 'call',
        n_steps: int = 100,
        n_paths: int = 10000
    ) -> float:
        """
        Price a European option using Monte Carlo with local volatility.

        Simulates: dS_t = (r-q) S_t dt + σ_loc(S_t, t) S_t dW_t

        Args:
            strike: Option strike price
            maturity: Time to maturity
            option_type: 'call' or 'put'
            n_steps: Number of time steps
            n_paths: Number of Monte Carlo paths

        Returns:
            Option price
        """
        dt = maturity / n_steps
        sqrt_dt = np.sqrt(dt)

        S = np.full(n_paths, self.spot)

        for step in range(n_steps):
            t = step * dt
            local_vol = np.array([self.get_local_volatility(s, t) for s in S])
            dW = np.random.standard_normal(n_paths)
            drift = (self.rate - self.dividend_yield) * S * dt
            diffusion = local_vol * S * sqrt_dt * dW
            S = S + drift + diffusion
            S = np.maximum(S, 1e-6)

        if option_type == 'call':
            payoff = np.maximum(S - strike, 0)
        else:
            payoff = np.maximum(strike - S, 0)

        return np.exp(-self.rate * maturity) * np.mean(payoff)

    def implied_volatility_from_local(
        self,
        strike: float,
        maturity: float,
        n_paths: int = 50000
    ) -> float:
        """
        Compute implied volatility from local volatility using Monte Carlo.

        This verifies the consistency between local and implied vol surfaces.

        Args:
            strike: Option strike
            maturity: Time to maturity
            n_paths: Number of MC paths

        Returns:
            Implied volatility
        """
        mc_price = self.price_european_option(strike, maturity, 'call', n_paths=n_paths)

        def price_diff(sigma):
            return self._black_scholes_call(strike, maturity, sigma) - mc_price

        try:
            implied_vol = brentq(price_diff, 0.001, 3.0)
        except ValueError:
            implied_vol = self._implied_vols[
                np.argmin(np.abs(self._strikes - strike)),
                np.argmin(np.abs(self._maturities - maturity))
            ]

        return implied_vol

    @property
    def strikes(self) -> np.ndarray:
        """Get calibration strikes."""
        return self._strikes

    @property
    def maturities(self) -> np.ndarray:
        """Get calibration maturities."""
        return self._maturities

    @property
    def local_vols(self) -> np.ndarray:
        """Get local volatility matrix."""
        return self._local_vols

    @property
    def implied_vols(self) -> np.ndarray:
        """Get implied volatility matrix."""
        return self._implied_vols


@jit(nopython=True)
def dupire_local_var_fast(
    dC_dT: float,
    dC_dK: float,
    d2C_dK2: float,
    K: float,
    r: float,
    q: float,
    C: float
) -> float:
    """
    Fast Numba-compiled Dupire local variance calculation.

    σ²_loc = [∂C/∂T + (r-q)K·∂C/∂K + q·C] / [0.5·K²·∂²C/∂K²]
    """
    numerator = dC_dT + (r - q) * K * dC_dK + q * C
    denominator = 0.5 * K * K * d2C_dK2

    if denominator <= 1e-10:
        return 0.04

    local_var = numerator / denominator

    if local_var <= 0:
        return 0.04

    return local_var
