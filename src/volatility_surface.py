"""
Volatility Surface Base Class

Provides the foundation for volatility surface modeling with support for:
- Sticky Strike parameterization
- Sticky Delta parameterization
- Surface interpolation and extrapolation
"""

import numpy as np
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline, interp1d
from typing import Optional, Literal, Tuple, Union
from dataclasses import dataclass


@dataclass
class VolatilitySurfaceData:
    """Container for volatility surface data."""
    strikes: np.ndarray
    maturities: np.ndarray
    implied_vols: np.ndarray
    spot: float
    rate: float = 0.0
    dividend_yield: float = 0.0


class VolatilitySurface:
    """
    Volatility Surface with Sticky Strike and Sticky Delta Parameterization.

    Mathematical Background:
    ------------------------
    A volatility surface σ(K, T) provides implied volatilities across different
    strikes K and maturities T. The surface must satisfy no-arbitrage constraints.

    Sticky Strike:
        The volatility surface is parameterized based on absolute strike levels.
        σ(K, T) remains fixed for specific strikes even as spot S changes.
        This implies the smile "sticks" to the same strike values.

    Sticky Delta:
        The volatility surface is parameterized based on delta levels.
        σ(Δ, T) where Δ = ∂C/∂S is the option delta.
        This approach is more dynamic, adjusting volatilities based on the
        relative position of the strike to spot price (moneyness).

        For a call option: Δ_call = e^(-qT) * N(d1)
        For a put option:  Δ_put = -e^(-qT) * N(-d1)

        where d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)

    Attributes:
        spot: Current spot price
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        parameterization: 'strike' or 'delta'
    """

    def __init__(
        self,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
        parameterization: Literal['strike', 'delta'] = 'strike'
    ):
        """
        Initialize the volatility surface.

        Args:
            spot: Current spot price of the underlying
            rate: Risk-free interest rate (annualized)
            dividend_yield: Continuous dividend yield (annualized)
            parameterization: 'strike' for Sticky Strike, 'delta' for Sticky Delta
        """
        self.spot = spot
        self.rate = rate
        self.dividend_yield = dividend_yield
        self.parameterization = parameterization

        self._strikes: Optional[np.ndarray] = None
        self._maturities: Optional[np.ndarray] = None
        self._implied_vols: Optional[np.ndarray] = None
        self._deltas: Optional[np.ndarray] = None
        self._interpolator: Optional[RectBivariateSpline] = None

    def set_data(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray
    ) -> None:
        """
        Set the volatility surface data.

        Args:
            strikes: Array of strike prices (shape: n_strikes,)
            maturities: Array of maturities in years (shape: n_maturities,)
            implied_vols: 2D array of implied volatilities
                         (shape: n_strikes x n_maturities)
        """
        self._strikes = np.asarray(strikes)
        self._maturities = np.asarray(maturities)
        self._implied_vols = np.asarray(implied_vols)

        if self._implied_vols.shape != (len(self._strikes), len(self._maturities)):
            raise ValueError(
                f"implied_vols shape {self._implied_vols.shape} must match "
                f"({len(self._strikes)}, {len(self._maturities)})"
            )

        self._build_interpolator()

        if self.parameterization == 'delta':
            self._compute_deltas()

    def _build_interpolator(self) -> None:
        """Build the bivariate spline interpolator."""
        self._interpolator = RectBivariateSpline(
            self._strikes,
            self._maturities,
            self._implied_vols,
            kx=min(3, len(self._strikes) - 1),
            ky=min(3, len(self._maturities) - 1)
        )

    def _compute_deltas(self) -> None:
        """Compute delta values for each strike/maturity combination."""
        self._deltas = np.zeros_like(self._implied_vols)

        for i, K in enumerate(self._strikes):
            for j, T in enumerate(self._maturities):
                sigma = self._implied_vols[i, j]
                self._deltas[i, j] = self._black_scholes_delta(K, T, sigma)

    def _black_scholes_delta(
        self,
        strike: float,
        maturity: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate Black-Scholes delta.

        Δ_call = e^(-qT) * N(d1)
        Δ_put = -e^(-qT) * N(-d1)

        where d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
        """
        if maturity <= 0 or sigma <= 0:
            return 0.0

        S, K, r, q, T = self.spot, strike, self.rate, self.dividend_yield, maturity

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        if option_type == 'call':
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return -np.exp(-q * T) * norm.cdf(-d1)

    def _black_scholes_price(
        self,
        strike: float,
        maturity: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate Black-Scholes option price.

        C(S,T) = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)
        P(S,T) = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)
        """
        if maturity <= 0:
            if option_type == 'call':
                return max(self.spot - strike, 0)
            else:
                return max(strike - self.spot, 0)

        S, K, r, q, T = self.spot, strike, self.rate, self.dividend_yield, maturity

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    def get_implied_vol(
        self,
        strike: float,
        maturity: float,
        delta: Optional[float] = None
    ) -> float:
        """
        Get interpolated implied volatility.

        Args:
            strike: Strike price (used if parameterization='strike')
            maturity: Time to maturity in years
            delta: Option delta (used if parameterization='delta')

        Returns:
            Interpolated implied volatility
        """
        if self._interpolator is None:
            raise ValueError("Surface data not set. Call set_data() first.")

        if self.parameterization == 'strike':
            return float(self._interpolator(strike, maturity)[0, 0])
        else:
            if delta is None:
                raise ValueError("Delta required for delta parameterization")
            strike = self._delta_to_strike(delta, maturity)
            return float(self._interpolator(strike, maturity)[0, 0])

    def _delta_to_strike(self, delta: float, maturity: float) -> float:
        """
        Convert delta to strike price using Newton-Raphson iteration.

        Given Δ and T, solve for K in:
        Δ = e^(-qT) * N(d1)
        """
        from scipy.optimize import brentq

        sigma_atm = self.get_atm_vol(maturity)

        def delta_diff(K):
            sigma = float(self._interpolator(K, maturity)[0, 0])
            return self._black_scholes_delta(K, maturity, sigma) - delta

        K_min = self.spot * 0.5
        K_max = self.spot * 2.0

        try:
            return brentq(delta_diff, K_min, K_max)
        except ValueError:
            return self.spot

    def get_atm_vol(self, maturity: float) -> float:
        """Get at-the-money implied volatility for a given maturity."""
        forward = self.spot * np.exp((self.rate - self.dividend_yield) * maturity)
        return self.get_implied_vol(forward, maturity)

    def get_moneyness(self, strike: float, maturity: float) -> float:
        """
        Calculate log-moneyness.

        m = ln(K/F) where F = S*e^((r-q)T)
        """
        forward = self.spot * np.exp((self.rate - self.dividend_yield) * maturity)
        return np.log(strike / forward)

    def get_standardized_moneyness(self, strike: float, maturity: float) -> float:
        """
        Calculate standardized moneyness (used in SVI parameterization).

        k = ln(K/F) / σ_ATM√T
        """
        moneyness = self.get_moneyness(strike, maturity)
        atm_vol = self.get_atm_vol(maturity)
        return moneyness / (atm_vol * np.sqrt(maturity))

    def total_variance(self, strike: float, maturity: float) -> float:
        """
        Calculate total implied variance.

        w(K, T) = σ²(K, T) * T
        """
        vol = self.get_implied_vol(strike, maturity)
        return vol**2 * maturity

    def shift_spot(self, new_spot: float) -> None:
        """
        Update the spot price (behavior depends on parameterization).

        Sticky Strike: Surface remains unchanged
        Sticky Delta: Surface shifts with spot
        """
        old_spot = self.spot
        self.spot = new_spot

        if self.parameterization == 'delta' and self._deltas is not None:
            self._compute_deltas()

    def get_smile(self, maturity: float, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the volatility smile for a given maturity.

        Args:
            maturity: Time to maturity
            n_points: Number of points in the smile

        Returns:
            Tuple of (strikes, implied_vols)
        """
        forward = self.spot * np.exp((self.rate - self.dividend_yield) * maturity)
        strikes = np.linspace(0.5 * forward, 1.5 * forward, n_points)
        vols = np.array([self.get_implied_vol(K, maturity) for K in strikes])
        return strikes, vols

    def get_term_structure(self, strike: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the volatility term structure for a given strike (or ATM).

        Args:
            strike: Strike price (None for ATM)

        Returns:
            Tuple of (maturities, implied_vols)
        """
        if self._maturities is None:
            raise ValueError("Surface data not set.")

        maturities = self._maturities

        if strike is None:
            vols = np.array([self.get_atm_vol(T) for T in maturities])
        else:
            vols = np.array([self.get_implied_vol(strike, T) for T in maturities])

        return maturities, vols

    @property
    def strikes(self) -> np.ndarray:
        """Get the strike array."""
        return self._strikes

    @property
    def maturities(self) -> np.ndarray:
        """Get the maturity array."""
        return self._maturities

    @property
    def implied_vols(self) -> np.ndarray:
        """Get the implied volatility matrix."""
        return self._implied_vols


def create_sample_surface(
    spot: float = 100.0,
    rate: float = 0.05,
    base_vol: float = 0.20,
    skew: float = -0.1,
    term_slope: float = 0.02
) -> VolatilitySurface:
    """
    Create a sample volatility surface for testing.

    Args:
        spot: Spot price
        rate: Risk-free rate
        base_vol: At-the-money volatility
        skew: Volatility skew parameter
        term_slope: Term structure slope

    Returns:
        VolatilitySurface with sample data
    """
    strikes = np.linspace(0.7 * spot, 1.3 * spot, 15)
    maturities = np.array([0.083, 0.25, 0.5, 1.0, 2.0])

    implied_vols = np.zeros((len(strikes), len(maturities)))

    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            moneyness = np.log(K / spot)
            vol = base_vol + skew * moneyness + term_slope * np.sqrt(T)
            vol += 0.05 * moneyness**2
            implied_vols[i, j] = max(0.01, vol)

    surface = VolatilitySurface(spot, rate)
    surface.set_data(strikes, maturities, implied_vols)

    return surface
