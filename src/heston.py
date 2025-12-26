"""
Heston Stochastic Volatility Model

The Heston model is one of the most widely used stochastic volatility models,
featuring mean-reverting variance and correlation between asset and variance processes.
"""

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from numba import jit, complex128, float64


@dataclass
class HestonParameters:
    """
    Heston Model Parameters.

    v0: Initial variance (σ²_0)
    kappa: Mean reversion speed of variance
    theta: Long-term variance mean (σ²_∞)
    sigma: Volatility of variance (vol of vol)
    rho: Correlation between asset and variance Wiener processes
    """
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def __post_init__(self):
        """Validate Feller condition for non-negative variance."""
        feller = 2 * self.kappa * self.theta - self.sigma**2
        if feller < 0:
            import warnings
            warnings.warn(
                f"Feller condition violated: 2κθ - σ² = {feller:.4f} < 0. "
                "Variance process may hit zero."
            )

    @property
    def feller_ratio(self) -> float:
        """Feller ratio: should be > 1 for variance to stay positive."""
        return 2 * self.kappa * self.theta / (self.sigma**2)


class HestonModel:
    """
    Heston Stochastic Volatility Model Implementation.

    Mathematical Framework:
    -----------------------
    The Heston model describes the evolution of the asset price S and its
    variance v with the following system of SDEs:

        dS_t = r·S_t·dt + √v_t·S_t·dW^S_t

        dv_t = κ(θ - v_t)dt + σ·√v_t·dW^v_t

    The CRITICAL third equation is the correlation between the two Wiener processes:

        dW^S_t · dW^v_t = ρ·dt

    or equivalently: E[dW^S_t · dW^v_t] = ρ·dt

    Parameters:
    -----------
    - r: Risk-free rate
    - v_0: Initial variance
    - κ (kappa): Mean reversion speed of variance
    - θ (theta): Long-term variance level
    - σ (sigma): Volatility of variance (vol-of-vol)
    - ρ (rho): Correlation between S and v

    The correlation ρ is crucial as it:
    - Controls the skew of the implied volatility smile
    - Negative ρ produces negative skew (leverage effect)
    - Positive ρ produces positive skew

    Feller Condition:
    -----------------
    For variance to remain strictly positive: 2κθ > σ²

    Characteristic Function:
    ------------------------
    The Heston model has a semi-closed form solution using its characteristic
    function, which allows efficient option pricing via Fourier inversion.
    """

    def __init__(
        self,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0
    ):
        """
        Initialize the Heston model.

        Args:
            spot: Current spot price
            rate: Risk-free interest rate
            dividend_yield: Continuous dividend yield
        """
        self.spot = spot
        self.rate = rate
        self.dividend_yield = dividend_yield
        self.params: Optional[HestonParameters] = None

    def set_parameters(
        self,
        v0: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float
    ) -> None:
        """
        Set Heston model parameters.

        Args:
            v0: Initial variance
            kappa: Mean reversion speed
            theta: Long-term variance
            sigma: Vol of vol
            rho: Correlation between Wiener processes
        """
        self.params = HestonParameters(v0, kappa, theta, sigma, rho)

    def characteristic_function(
        self,
        u: complex,
        maturity: float,
        variant: int = 1
    ) -> complex:
        """
        Heston characteristic function for log-price.

        φ(u) = E[exp(iu·ln(S_T))]

        Using the formulation from Gatheral (2006) with two variants
        to ensure numerical stability.

        Args:
            u: Complex frequency
            maturity: Time to maturity
            variant: 1 or 2 for different branch cuts

        Returns:
            Characteristic function value
        """
        if self.params is None:
            raise ValueError("Parameters not set. Call set_parameters() first.")

        return heston_char_func(
            u,
            maturity,
            self.rate,
            self.dividend_yield,
            self.params.v0,
            self.params.kappa,
            self.params.theta,
            self.params.sigma,
            self.params.rho,
            variant
        )

    def price_european_call(
        self,
        strike: float,
        maturity: float,
        integration_method: str = 'quad'
    ) -> float:
        """
        Price a European call option using the Heston semi-closed form.

        C(K,T) = S·e^(-qT)·P1 - K·e^(-rT)·P2

        where P1 and P2 are computed via Fourier inversion:

        P_j = 0.5 + (1/π) ∫_0^∞ Re[e^(-iu·ln(K))·φ_j(u) / (iu)] du

        Args:
            strike: Option strike price
            maturity: Time to maturity
            integration_method: 'quad' for scipy.integrate.quad

        Returns:
            Call option price
        """
        if self.params is None:
            raise ValueError("Parameters not set. Call set_parameters() first.")

        S = self.spot
        K = strike
        T = maturity
        r = self.rate
        q = self.dividend_yield

        log_K = np.log(K)

        def integrand_P1(u):
            """Integrand for P1."""
            phi = self.characteristic_function(u - 1j, T, variant=1)
            numerator = np.exp(-1j * u * log_K) * phi
            denominator = 1j * u * self.characteristic_function(-1j, T, variant=1)
            return np.real(numerator / denominator)

        def integrand_P2(u):
            """Integrand for P2."""
            phi = self.characteristic_function(u, T, variant=2)
            return np.real(np.exp(-1j * u * log_K) * phi / (1j * u))

        P1 = 0.5 + (1 / np.pi) * quad(integrand_P1, 0, 100, limit=200)[0]
        P2 = 0.5 + (1 / np.pi) * quad(integrand_P2, 0, 100, limit=200)[0]

        call_price = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

        return max(call_price, 0)

    def price_european_put(self, strike: float, maturity: float) -> float:
        """
        Price a European put option using put-call parity.

        P = C - S·e^(-qT) + K·e^(-rT)
        """
        call_price = self.price_european_call(strike, maturity)
        forward_value = self.spot * np.exp(-self.dividend_yield * maturity)
        discount = np.exp(-self.rate * maturity)

        return call_price - forward_value + strike * discount

    def implied_volatility(
        self,
        strike: float,
        maturity: float,
        option_type: str = 'call'
    ) -> float:
        """
        Compute implied volatility from Heston price.

        Args:
            strike: Option strike
            maturity: Time to maturity
            option_type: 'call' or 'put'

        Returns:
            Black-Scholes implied volatility
        """
        from scipy.optimize import brentq

        if option_type == 'call':
            target_price = self.price_european_call(strike, maturity)
        else:
            target_price = self.price_european_put(strike, maturity)

        def price_diff(sigma):
            return self._black_scholes_price(strike, maturity, sigma, option_type) - target_price

        try:
            return brentq(price_diff, 0.001, 3.0)
        except ValueError:
            return np.sqrt(self.params.theta)

    def _black_scholes_price(
        self,
        strike: float,
        maturity: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        """Calculate Black-Scholes option price."""
        S, K, r, q, T = self.spot, strike, self.rate, self.dividend_yield, maturity

        if T <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    def generate_implied_vol_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray
    ) -> np.ndarray:
        """
        Generate implied volatility surface from Heston model.

        Args:
            strikes: Array of strike prices
            maturities: Array of maturities

        Returns:
            2D array of implied volatilities (strikes x maturities)
        """
        implied_vols = np.zeros((len(strikes), len(maturities)))

        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                implied_vols[i, j] = self.implied_volatility(K, T)

        return implied_vols

    def calibrate(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_prices: np.ndarray,
        option_types: Optional[np.ndarray] = None,
        method: str = 'differential_evolution'
    ) -> HestonParameters:
        """
        Calibrate Heston parameters to market prices.

        Minimizes the sum of squared pricing errors.

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            market_prices: Array of market option prices
            option_types: Array of 'call'/'put' (default all calls)
            method: Optimization method

        Returns:
            Calibrated HestonParameters
        """
        if option_types is None:
            option_types = np.array(['call'] * len(strikes))

        def objective(params):
            v0, kappa, theta, sigma, rho = params

            if 2 * kappa * theta < 0.1 * sigma**2:
                return 1e10

            self.set_parameters(v0, kappa, theta, sigma, rho)

            total_error = 0
            for K, T, market_price, opt_type in zip(strikes, maturities, market_prices, option_types):
                try:
                    if opt_type == 'call':
                        model_price = self.price_european_call(K, T)
                    else:
                        model_price = self.price_european_put(K, T)
                    total_error += (model_price - market_price)**2
                except:
                    total_error += 1e10

            return total_error

        bounds = [
            (0.001, 1.0),
            (0.1, 10.0),
            (0.001, 1.0),
            (0.01, 2.0),
            (-0.99, 0.99)
        ]

        if method == 'differential_evolution':
            result = differential_evolution(objective, bounds, maxiter=200, seed=42)
        else:
            x0 = [0.04, 2.0, 0.04, 0.3, -0.7]
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        v0, kappa, theta, sigma, rho = result.x
        self.set_parameters(v0, kappa, theta, sigma, rho)

        return self.params

    def simulate_paths(
        self,
        maturity: float,
        n_paths: int = 10000,
        n_steps: int = 252,
        scheme: str = 'euler'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate price and variance paths using Monte Carlo.

        Simulates the correlated SDEs:
            dS_t = r·S_t·dt + √v_t·S_t·dW^S_t
            dv_t = κ(θ - v_t)dt + σ·√v_t·dW^v_t
            dW^S · dW^v = ρ·dt

        Args:
            maturity: Simulation time horizon
            n_paths: Number of paths
            n_steps: Number of time steps
            scheme: 'euler' or 'milstein'

        Returns:
            Tuple of (price_paths, variance_paths) with shape (n_steps+1, n_paths)
        """
        if self.params is None:
            raise ValueError("Parameters not set.")

        dt = maturity / n_steps
        sqrt_dt = np.sqrt(dt)

        S = np.zeros((n_steps + 1, n_paths))
        v = np.zeros((n_steps + 1, n_paths))

        S[0] = self.spot
        v[0] = self.params.v0

        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma
        rho = self.params.rho
        r = self.rate
        q = self.dividend_yield

        for t in range(n_steps):
            Z1 = np.random.standard_normal(n_paths)
            Z2 = np.random.standard_normal(n_paths)

            W_S = Z1
            W_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2

            v_current = np.maximum(v[t], 0)
            sqrt_v = np.sqrt(v_current)

            S[t + 1] = S[t] * np.exp(
                (r - q - 0.5 * v_current) * dt + sqrt_v * sqrt_dt * W_S
            )

            if scheme == 'milstein':
                v[t + 1] = (
                    v_current
                    + kappa * (theta - v_current) * dt
                    + sigma * sqrt_v * sqrt_dt * W_v
                    + 0.25 * sigma**2 * dt * (W_v**2 - 1)
                )
            else:
                v[t + 1] = (
                    v_current
                    + kappa * (theta - v_current) * dt
                    + sigma * sqrt_v * sqrt_dt * W_v
                )

            v[t + 1] = np.maximum(v[t + 1], 0)

        return S, v

    def price_european_mc(
        self,
        strike: float,
        maturity: float,
        option_type: str = 'call',
        n_paths: int = 100000
    ) -> Tuple[float, float]:
        """
        Price European option using Monte Carlo simulation.

        Args:
            strike: Option strike
            maturity: Time to maturity
            option_type: 'call' or 'put'
            n_paths: Number of MC paths

        Returns:
            Tuple of (price, standard_error)
        """
        S, _ = self.simulate_paths(maturity, n_paths)

        S_T = S[-1]

        if option_type == 'call':
            payoffs = np.maximum(S_T - strike, 0)
        else:
            payoffs = np.maximum(strike - S_T, 0)

        discount = np.exp(-self.rate * maturity)
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(n_paths)

        return price, std_error


def heston_char_func(
    u: complex,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    variant: int = 1
) -> complex:
    """
    Heston characteristic function (optimized).

    φ(u) = exp(C + D·v0 + iu·ln(S) + iu·(r-q)·T)

    where C and D are complex functions of the parameters.
    """
    i = 1j

    if variant == 1:
        b = kappa - rho * sigma
        u_adj = u - i
    else:
        b = kappa
        u_adj = u

    a = kappa * theta

    d = np.sqrt((rho * sigma * i * u_adj - b)**2 + sigma**2 * (i * u_adj + u_adj**2))

    g = (b - rho * sigma * i * u_adj - d) / (b - rho * sigma * i * u_adj + d)

    exp_neg_dT = np.exp(-d * T)

    C = (r - q) * i * u * T + (a / sigma**2) * (
        (b - rho * sigma * i * u_adj - d) * T
        - 2 * np.log((1 - g * exp_neg_dT) / (1 - g))
    )

    D = ((b - rho * sigma * i * u_adj - d) / sigma**2) * (
        (1 - exp_neg_dT) / (1 - g * exp_neg_dT)
    )

    return np.exp(C + D * v0)
