"""
Interpolation and Smoothing Techniques for Volatility Surfaces

This module provides various methods for interpolating and extrapolating
volatility surfaces while maintaining smoothness and arbitrage-free properties.
"""

import numpy as np
from scipy.interpolate import (
    RectBivariateSpline,
    interp1d,
    UnivariateSpline,
    CubicSpline
)
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from typing import Optional, Tuple, Literal, Callable
from dataclasses import dataclass


@dataclass
class InterpolationConfig:
    """Configuration for surface interpolation."""
    method: str = 'cubic_spline'
    smoothing: float = 0.0
    extrapolation: str = 'flat'
    kernel_bandwidth: float = 0.1


class SurfaceInterpolator:
    """
    Volatility Surface Interpolation and Smoothing.

    This class provides various methods for:
    1. Interpolation between grid points
    2. Smoothing noisy market data
    3. Extrapolation to extreme strikes/maturities

    Interpolation Methods:
    ----------------------
    1. CUBIC SPLINES
       - Natural cubic splines provide C² continuity
       - Good for smooth surfaces with sufficient data
       - Can oscillate near boundaries

    2. BIVARIATE SPLINES (RectBivariateSpline)
       - 2D tensor product splines
       - Efficient for regular grids
       - Supports smoothing parameter

    3. GAUSSIAN KERNEL SMOOTHING
       - Non-parametric smoothing
       - Bandwidth controls smoothness
       - Robust to outliers

    4. TOTAL VARIANCE INTERPOLATION
       - Interpolates w = σ²T instead of σ
       - Naturally preserves calendar spread arbitrage
       - Standard in practice

    Extrapolation Methods:
    ----------------------
    1. FLAT EXTRAPOLATION
       - Extend boundary values
       - Simple but may not capture wing behavior

    2. LOG-LINEAR EXTRAPOLATION
       - Linear in log-strike space
       - Better for extreme strikes

    3. ASYMPTOTIC MODELS
       - SVI for strike extrapolation
       - Power-law for maturity extrapolation

    4. LEE'S MOMENT FORMULA
       - For extreme strike behavior:
         σ²(k)T → 2|k| as |k| → ∞
       - Controls wing slopes
    """

    def __init__(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0
    ):
        """
        Initialize the surface interpolator.

        Args:
            strikes: Array of strike prices
            maturities: Array of maturities
            implied_vols: 2D array of implied volatilities (strikes x maturities)
            spot: Current spot price
            rate: Risk-free interest rate
            dividend_yield: Continuous dividend yield
        """
        self.strikes = np.asarray(strikes)
        self.maturities = np.asarray(maturities)
        self.implied_vols = np.asarray(implied_vols)
        self.spot = spot
        self.rate = rate
        self.dividend_yield = dividend_yield

        self._total_variance = self.implied_vols**2 * self.maturities[np.newaxis, :]

        self._interpolator: Optional[RectBivariateSpline] = None
        self._smile_interpolators: dict = {}

    def build_interpolator(
        self,
        method: str = 'cubic_spline',
        smoothing: float = 0.0
    ) -> None:
        """
        Build the surface interpolator.

        Args:
            method: 'cubic_spline', 'linear', or 'total_variance'
            smoothing: Smoothing parameter (0 = exact interpolation)
        """
        if method == 'total_variance':
            self._interpolator = RectBivariateSpline(
                self.strikes,
                self.maturities,
                self._total_variance,
                kx=min(3, len(self.strikes) - 1),
                ky=min(3, len(self.maturities) - 1),
                s=smoothing
            )
            self._interp_variance = True
        else:
            k = 3 if method == 'cubic_spline' else 1
            self._interpolator = RectBivariateSpline(
                self.strikes,
                self.maturities,
                self.implied_vols,
                kx=min(k, len(self.strikes) - 1),
                ky=min(k, len(self.maturities) - 1),
                s=smoothing
            )
            self._interp_variance = False

    def interpolate(
        self,
        strike: float,
        maturity: float,
        extrapolation: str = 'flat'
    ) -> float:
        """
        Interpolate/extrapolate implied volatility at (strike, maturity).

        Args:
            strike: Strike price
            maturity: Time to maturity
            extrapolation: 'flat', 'linear', 'log_linear', or 'asymptotic'

        Returns:
            Interpolated implied volatility
        """
        if self._interpolator is None:
            self.build_interpolator()

        K_min, K_max = self.strikes[0], self.strikes[-1]
        T_min, T_max = self.maturities[0], self.maturities[-1]

        K_in_range = K_min <= strike <= K_max
        T_in_range = T_min <= maturity <= T_max

        if K_in_range and T_in_range:
            if self._interp_variance:
                w = float(self._interpolator(strike, maturity)[0, 0])
                return np.sqrt(w / maturity)
            else:
                return float(self._interpolator(strike, maturity)[0, 0])

        return self._extrapolate(strike, maturity, extrapolation)

    def _extrapolate(
        self,
        strike: float,
        maturity: float,
        method: str = 'flat'
    ) -> float:
        """
        Extrapolate implied volatility outside the grid.

        Args:
            strike: Strike price
            maturity: Time to maturity
            method: Extrapolation method

        Returns:
            Extrapolated implied volatility
        """
        K_min, K_max = self.strikes[0], self.strikes[-1]
        T_min, T_max = self.maturities[0], self.maturities[-1]

        K_clamped = np.clip(strike, K_min, K_max)
        T_clamped = np.clip(maturity, T_min, T_max)

        if method == 'flat':
            if self._interp_variance:
                w = float(self._interpolator(K_clamped, T_clamped)[0, 0])
                w_scaled = w * (maturity / T_clamped) if T_clamped > 0 else w
                return np.sqrt(w_scaled / maturity)
            else:
                return float(self._interpolator(K_clamped, T_clamped)[0, 0])

        elif method == 'linear':
            base_vol = float(self._interpolator(K_clamped, T_clamped)[0, 0])

            if strike < K_min:
                dK = K_min - strike
                vol_slope = (self.implied_vols[1, :].mean() - self.implied_vols[0, :].mean()) / (self.strikes[1] - self.strikes[0])
                return base_vol - vol_slope * dK
            elif strike > K_max:
                dK = strike - K_max
                vol_slope = (self.implied_vols[-1, :].mean() - self.implied_vols[-2, :].mean()) / (self.strikes[-1] - self.strikes[-2])
                return base_vol + vol_slope * dK
            else:
                return base_vol

        elif method == 'log_linear':
            return self._log_linear_extrapolate(strike, maturity)

        elif method == 'asymptotic':
            return self._asymptotic_extrapolate(strike, maturity)

        else:
            raise ValueError(f"Unknown extrapolation method: {method}")

    def _log_linear_extrapolate(
        self,
        strike: float,
        maturity: float
    ) -> float:
        """
        Log-linear extrapolation in strike space.

        Assumes log(σ) is linear in log(K) for extreme strikes.
        """
        T_clamped = np.clip(maturity, self.maturities[0], self.maturities[-1])
        T_idx = np.argmin(np.abs(self.maturities - T_clamped))

        smile = self.implied_vols[:, T_idx]
        log_strikes = np.log(self.strikes)
        log_vols = np.log(smile)

        if strike < self.strikes[0]:
            slope = (log_vols[1] - log_vols[0]) / (log_strikes[1] - log_strikes[0])
            log_vol = log_vols[0] + slope * (np.log(strike) - log_strikes[0])
        else:
            slope = (log_vols[-1] - log_vols[-2]) / (log_strikes[-1] - log_strikes[-2])
            log_vol = log_vols[-1] + slope * (np.log(strike) - log_strikes[-1])

        return np.exp(log_vol)

    def _asymptotic_extrapolate(
        self,
        strike: float,
        maturity: float
    ) -> float:
        """
        Asymptotic extrapolation using Lee's moment formula.

        For extreme strikes, total variance behaves as:
            w(k) → β|k| as |k| → ∞

        where k = log(K/F) and β ≤ 2 (Lee's bound).
        """
        forward = self.spot * np.exp((self.rate - self.dividend_yield) * maturity)
        k = np.log(strike / forward)

        T_clamped = np.clip(maturity, self.maturities[0], self.maturities[-1])
        atm_idx = np.argmin(np.abs(self.strikes - forward))
        T_idx = np.argmin(np.abs(self.maturities - T_clamped))

        atm_vol = self.implied_vols[atm_idx, T_idx]
        atm_var = atm_vol**2 * T_clamped

        lee_slope = min(1.5, atm_var / (T_clamped * 0.5))

        if abs(k) > 0.5:
            boundary_k = 0.5 * np.sign(k)
            boundary_strike = forward * np.exp(boundary_k)
            boundary_strike = np.clip(boundary_strike, self.strikes[0], self.strikes[-1])
            boundary_var = float(self._interpolator(boundary_strike, T_clamped)[0, 0])**2 * T_clamped

            total_var = boundary_var + lee_slope * (abs(k) - 0.5) * maturity
        else:
            K_clamped = np.clip(strike, self.strikes[0], self.strikes[-1])
            if self._interp_variance:
                total_var = float(self._interpolator(K_clamped, T_clamped)[0, 0])
            else:
                vol = float(self._interpolator(K_clamped, T_clamped)[0, 0])
                total_var = vol**2 * maturity

        return np.sqrt(total_var / maturity)

    def smooth_surface(
        self,
        method: str = 'gaussian',
        **kwargs
    ) -> np.ndarray:
        """
        Apply smoothing to the volatility surface.

        Args:
            method: 'gaussian', 'spline', or 'kernel'
            **kwargs: Method-specific parameters

        Returns:
            Smoothed implied volatility surface
        """
        if method == 'gaussian':
            return self._gaussian_smooth(**kwargs)
        elif method == 'spline':
            return self._spline_smooth(**kwargs)
        elif method == 'kernel':
            return self._kernel_smooth(**kwargs)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    def _gaussian_smooth(
        self,
        sigma_strike: float = 1.0,
        sigma_time: float = 1.0
    ) -> np.ndarray:
        """
        Apply Gaussian kernel smoothing.

        Args:
            sigma_strike: Smoothing in strike direction
            sigma_time: Smoothing in time direction

        Returns:
            Smoothed surface
        """
        return gaussian_filter(
            self.implied_vols,
            sigma=(sigma_strike, sigma_time),
            mode='nearest'
        )

    def _spline_smooth(
        self,
        smoothing: float = 0.1
    ) -> np.ndarray:
        """
        Apply spline smoothing.

        Args:
            smoothing: Smoothing parameter

        Returns:
            Smoothed surface
        """
        smoother = RectBivariateSpline(
            self.strikes,
            self.maturities,
            self.implied_vols,
            s=smoothing * len(self.strikes) * len(self.maturities)
        )

        smoothed = np.zeros_like(self.implied_vols)
        for i, K in enumerate(self.strikes):
            for j, T in enumerate(self.maturities):
                smoothed[i, j] = smoother(K, T)[0, 0]

        return smoothed

    def _kernel_smooth(
        self,
        bandwidth_strike: float = 0.1,
        bandwidth_time: float = 0.1,
        kernel: str = 'gaussian'
    ) -> np.ndarray:
        """
        Apply Nadaraya-Watson kernel smoothing.

        σ_smooth(K, T) = Σ w_i σ_i / Σ w_i

        where w_i = K_h((K - K_i)/h_K) × K_h((T - T_i)/h_T)

        Args:
            bandwidth_strike: Bandwidth in strike direction
            bandwidth_time: Bandwidth in time direction
            kernel: 'gaussian' or 'epanechnikov'

        Returns:
            Smoothed surface
        """
        if kernel == 'gaussian':
            kernel_func = lambda x: np.exp(-0.5 * x**2)
        else:
            kernel_func = lambda x: np.maximum(0, 0.75 * (1 - x**2))

        K_range = self.strikes[-1] - self.strikes[0]
        T_range = self.maturities[-1] - self.maturities[0]

        h_K = bandwidth_strike * K_range
        h_T = bandwidth_time * T_range

        smoothed = np.zeros_like(self.implied_vols)

        for i, K in enumerate(self.strikes):
            for j, T in enumerate(self.maturities):
                weights = np.zeros_like(self.implied_vols)
                for ii, Ki in enumerate(self.strikes):
                    for jj, Tj in enumerate(self.maturities):
                        w_K = kernel_func((K - Ki) / h_K) if h_K > 0 else 1.0
                        w_T = kernel_func((T - Tj) / h_T) if h_T > 0 else 1.0
                        weights[ii, jj] = w_K * w_T

                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    smoothed[i, j] = np.sum(weights * self.implied_vols) / weight_sum
                else:
                    smoothed[i, j] = self.implied_vols[i, j]

        return smoothed

    def fit_smile_spline(
        self,
        maturity: float,
        smoothing: float = 0.0
    ) -> UnivariateSpline:
        """
        Fit a univariate spline to the volatility smile at a given maturity.

        Args:
            maturity: Time to maturity
            smoothing: Smoothing parameter

        Returns:
            UnivariateSpline object for the smile
        """
        T_idx = np.argmin(np.abs(self.maturities - maturity))
        smile = self.implied_vols[:, T_idx]

        spline = UnivariateSpline(
            self.strikes,
            smile,
            k=min(3, len(self.strikes) - 1),
            s=smoothing
        )

        self._smile_interpolators[maturity] = spline
        return spline

    def fit_svi(
        self,
        maturity: float
    ) -> dict:
        """
        Fit SVI (Stochastic Volatility Inspired) parameterization to a smile.

        SVI raw parameterization:
            w(k) = a + b(ρ(k-m) + √((k-m)² + σ²))

        where:
            w = σ²T is total variance
            k = log(K/F) is log-moneyness
            a, b, ρ, m, σ are parameters

        Args:
            maturity: Time to maturity

        Returns:
            Dict of SVI parameters
        """
        T_idx = np.argmin(np.abs(self.maturities - maturity))
        smile = self.implied_vols[:, T_idx]
        total_var = smile**2 * maturity

        forward = self.spot * np.exp((self.rate - self.dividend_yield) * maturity)
        log_moneyness = np.log(self.strikes / forward)

        def svi_func(params, k):
            a, b, rho, m, sigma = params
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

        def objective(params):
            model_var = svi_func(params, log_moneyness)
            return np.sum((model_var - total_var)**2)

        atm_var = total_var[np.argmin(np.abs(log_moneyness))]
        x0 = [atm_var, 0.1, -0.3, 0.0, 0.1]

        bounds = [
            (0.001, 1.0),
            (0.001, 1.0),
            (-0.999, 0.999),
            (-0.5, 0.5),
            (0.001, 1.0)
        ]

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        a, b, rho, m, sigma = result.x
        return {
            'a': a,
            'b': b,
            'rho': rho,
            'm': m,
            'sigma': sigma,
            'maturity': maturity
        }

    def interpolate_smile(
        self,
        strike: float,
        maturity: float,
        method: str = 'spline'
    ) -> float:
        """
        Interpolate a single point using smile interpolation.

        Args:
            strike: Strike price
            maturity: Time to maturity
            method: 'spline' or 'linear'

        Returns:
            Interpolated implied volatility
        """
        T_lower_idx = np.searchsorted(self.maturities, maturity) - 1
        T_lower_idx = max(0, min(T_lower_idx, len(self.maturities) - 2))
        T_upper_idx = T_lower_idx + 1

        T_lower = self.maturities[T_lower_idx]
        T_upper = self.maturities[T_upper_idx]

        if method == 'spline':
            if T_lower not in self._smile_interpolators:
                self.fit_smile_spline(T_lower)
            if T_upper not in self._smile_interpolators:
                self.fit_smile_spline(T_upper)

            vol_lower = self._smile_interpolators[T_lower](strike)
            vol_upper = self._smile_interpolators[T_upper](strike)
        else:
            smile_lower = self.implied_vols[:, T_lower_idx]
            smile_upper = self.implied_vols[:, T_upper_idx]

            interp_lower = interp1d(self.strikes, smile_lower, fill_value='extrapolate')
            interp_upper = interp1d(self.strikes, smile_upper, fill_value='extrapolate')

            vol_lower = interp_lower(strike)
            vol_upper = interp_upper(strike)

        w_lower = vol_lower**2 * T_lower
        w_upper = vol_upper**2 * T_upper

        weight = (maturity - T_lower) / (T_upper - T_lower) if T_upper > T_lower else 0
        w_interp = w_lower + weight * (w_upper - w_lower)

        return np.sqrt(w_interp / maturity)

    def get_interpolated_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        method: str = 'bivariate'
    ) -> np.ndarray:
        """
        Get interpolated surface on a new grid.

        Args:
            strikes: New strike grid
            maturities: New maturity grid
            method: 'bivariate' or 'smile'

        Returns:
            2D array of interpolated implied volatilities
        """
        if self._interpolator is None:
            self.build_interpolator()

        surface = np.zeros((len(strikes), len(maturities)))

        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                if method == 'bivariate':
                    surface[i, j] = self.interpolate(K, T)
                else:
                    surface[i, j] = self.interpolate_smile(K, T)

        return surface
