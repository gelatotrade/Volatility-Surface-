"""
SABR Stochastic Alpha Beta Rho Model

The SABR model is widely used in interest rate derivatives and FX markets
for modeling the implied volatility smile. It operates in forward price space.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, brentq
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SABRParameters:
    """
    SABR Model Parameters.

    alpha: Initial volatility level (σ_0)
    beta: CEV exponent (0 ≤ β ≤ 1)
           β = 0: Normal SABR
           β = 1: Lognormal SABR
           0 < β < 1: CEV-like behavior
    rho: Correlation between forward and volatility processes
    nu: Volatility of volatility (vol of vol)
    """
    alpha: float
    beta: float
    rho: float
    nu: float

    def __post_init__(self):
        """Validate parameters."""
        if not 0 <= self.beta <= 1:
            raise ValueError(f"Beta must be in [0, 1], got {self.beta}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"Rho must be in [-1, 1], got {self.rho}")
        if self.alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {self.alpha}")
        if self.nu < 0:
            raise ValueError(f"Nu must be non-negative, got {self.nu}")


class SABRModel:
    """
    SABR (Stochastic Alpha Beta Rho) Model Implementation.

    Mathematical Framework:
    -----------------------
    The SABR model operates in FORWARD PRICE space (not spot price).
    This is crucial: the underlying is the forward price F, not spot S.

    The model dynamics are given by:

        dF_t = σ_t · F_t^β · dW_t

        dσ_t = ν · σ_t · dZ_t

    Note: There is NO DRIFT TERM in the forward price equation because
    F is already a martingale under the forward measure.

    The correlation between the two Wiener processes is:

        dW_t · dZ_t = ρ · dt

    Parameters:
    -----------
    - F_0: Initial forward price
    - α (alpha): Initial volatility σ_0
    - β (beta): CEV exponent controlling backbone of smile
    - ρ (rho): Correlation between F and σ
    - ν (nu): Volatility of volatility

    Implied Volatility Approximation:
    ---------------------------------
    The famous Hagan et al. (2002) approximation gives:

    σ_SABR(K, T) = α / [(FK)^((1-β)/2)] · [1 + ((1-β)²/24)·log²(F/K)/(FK)^(1-β) + ν²/(24α²(FK)^(1-β))] · [z/x(z)]

    where:
        z = (ν/α)(FK)^((1-β)/2) · log(F/K)
        x(z) = log[(√(1-2ρz+z²) + z - ρ) / (1-ρ)]

    For ATM options (F = K):
        σ_ATM = α / F^(1-β) · [1 + ((1-β)²α²/(24F^(2-2β)) + ρβνα/(4F^(1-β)) + (2-3ρ²)ν²/24) · T]

    Role of Parameters:
    -------------------
    - β: Controls the backbone slope of the smile
    - ρ: Controls the skew (negative ρ → negative skew)
    - ν: Controls the curvature/wings of the smile
    - α: Sets the overall level of volatility
    """

    def __init__(
        self,
        forward: float,
        maturity: float,
        rate: float = 0.0
    ):
        """
        Initialize the SABR model.

        Args:
            forward: Forward price (NOT spot price)
            maturity: Time to expiry
            rate: Risk-free rate (for discounting)
        """
        self.forward = forward
        self.maturity = maturity
        self.rate = rate
        self.params: Optional[SABRParameters] = None

    def set_parameters(
        self,
        alpha: float,
        beta: float,
        rho: float,
        nu: float
    ) -> None:
        """
        Set SABR model parameters.

        Args:
            alpha: Initial volatility
            beta: CEV exponent
            rho: Correlation
            nu: Vol of vol
        """
        self.params = SABRParameters(alpha, beta, rho, nu)

    def implied_volatility(self, strike: float) -> float:
        """
        Calculate SABR implied volatility using Hagan's approximation.

        The full formula:

        σ(K,F) = (α / (FK)^((1-β)/2)) · (1 + corrections) · (z/x(z))

        where the corrections account for higher-order terms.

        Args:
            strike: Option strike price

        Returns:
            SABR implied volatility
        """
        if self.params is None:
            raise ValueError("Parameters not set. Call set_parameters() first.")

        return sabr_implied_vol(
            strike,
            self.forward,
            self.maturity,
            self.params.alpha,
            self.params.beta,
            self.params.rho,
            self.params.nu
        )

    def implied_volatility_atm(self) -> float:
        """
        Calculate ATM implied volatility.

        For F = K:
        σ_ATM ≈ α/F^(1-β) · [1 + ((1-β)²α²/(24F^(2-2β)) + ρβνα/(4F^(1-β)) + (2-3ρ²)ν²/24)·T]
        """
        return self.implied_volatility(self.forward)

    def generate_smile(
        self,
        strikes: Optional[np.ndarray] = None,
        n_strikes: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the implied volatility smile.

        Args:
            strikes: Array of strikes (if None, auto-generated)
            n_strikes: Number of strikes if auto-generating

        Returns:
            Tuple of (strikes, implied_vols)
        """
        if strikes is None:
            K_min = 0.5 * self.forward
            K_max = 1.5 * self.forward
            strikes = np.linspace(K_min, K_max, n_strikes)

        vols = np.array([self.implied_volatility(K) for K in strikes])

        return strikes, vols

    def calibrate(
        self,
        strikes: np.ndarray,
        market_vols: np.ndarray,
        fix_beta: Optional[float] = None
    ) -> SABRParameters:
        """
        Calibrate SABR parameters to market implied volatilities.

        Minimizes sum of squared implied volatility errors.

        Args:
            strikes: Market strikes
            market_vols: Market implied volatilities
            fix_beta: If provided, fix beta at this value (common practice)

        Returns:
            Calibrated SABRParameters
        """
        if fix_beta is not None:
            x0 = [0.2, -0.3, 0.4]
            bounds = [(0.001, 2.0), (-0.999, 0.999), (0.001, 2.0)]

            def objective(params):
                alpha, rho, nu = params
                self.set_parameters(alpha, fix_beta, rho, nu)

                total_error = 0
                for K, market_vol in zip(strikes, market_vols):
                    try:
                        model_vol = self.implied_volatility(K)
                        total_error += (model_vol - market_vol)**2
                    except:
                        total_error += 1e10

                return total_error

            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            alpha, rho, nu = result.x
            self.set_parameters(alpha, fix_beta, rho, nu)

        else:
            x0 = [0.2, 0.5, -0.3, 0.4]
            bounds = [(0.001, 2.0), (0.0, 1.0), (-0.999, 0.999), (0.001, 2.0)]

            def objective(params):
                alpha, beta, rho, nu = params
                self.set_parameters(alpha, beta, rho, nu)

                total_error = 0
                for K, market_vol in zip(strikes, market_vols):
                    try:
                        model_vol = self.implied_volatility(K)
                        total_error += (model_vol - market_vol)**2
                    except:
                        total_error += 1e10

                return total_error

            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            alpha, beta, rho, nu = result.x
            self.set_parameters(alpha, beta, rho, nu)

        return self.params

    def calibrate_to_atm(self, atm_vol: float, beta: float = 0.5) -> None:
        """
        Quick calibration to match ATM volatility.

        Given ATM vol and beta, solve for alpha:
        α ≈ σ_ATM · F^(1-β)

        This is a first approximation; more accurate methods iterate.

        Args:
            atm_vol: Target ATM implied volatility
            beta: Fixed beta parameter
        """
        alpha_initial = atm_vol * (self.forward ** (1 - beta))

        def atm_diff(alpha):
            self.set_parameters(alpha, beta, 0.0, 0.3)
            return self.implied_volatility(self.forward) - atm_vol

        try:
            alpha = brentq(atm_diff, 0.001 * alpha_initial, 10 * alpha_initial)
        except ValueError:
            alpha = alpha_initial

        self.set_parameters(alpha, beta, 0.0, 0.3)

    def price_call(self, strike: float) -> float:
        """
        Price a European call option using SABR implied vol.

        C = e^(-rT) · [F·N(d1) - K·N(d2)]

        with σ = σ_SABR(K)

        Args:
            strike: Option strike

        Returns:
            Call option price
        """
        sigma = self.implied_volatility(strike)
        return self._black_call(strike, sigma)

    def price_put(self, strike: float) -> float:
        """
        Price a European put option using SABR implied vol.

        Args:
            strike: Option strike

        Returns:
            Put option price
        """
        sigma = self.implied_volatility(strike)
        return self._black_put(strike, sigma)

    def _black_call(self, strike: float, sigma: float) -> float:
        """Black's formula for call on forward."""
        F, K, T, r = self.forward, strike, self.maturity, self.rate

        if T <= 0 or sigma <= 0:
            return max(F - K, 0) * np.exp(-r * T)

        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

    def _black_put(self, strike: float, sigma: float) -> float:
        """Black's formula for put on forward."""
        F, K, T, r = self.forward, strike, self.maturity, self.rate

        if T <= 0 or sigma <= 0:
            return max(K - F, 0) * np.exp(-r * T)

        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    def local_volatility(self, strike: float) -> float:
        """
        Get the local volatility implied by SABR at strike K.

        For SABR: σ_loc(K) = α · K^(β-1)

        This is an approximation; full local vol requires Dupire.
        """
        if self.params is None:
            raise ValueError("Parameters not set.")

        return self.params.alpha * (strike ** (self.params.beta - 1))

    def delta(self, strike: float, option_type: str = 'call') -> float:
        """
        Calculate option delta.

        Δ = e^(-rT) · N(d1) for call
        Δ = -e^(-rT) · N(-d1) for put
        """
        sigma = self.implied_volatility(strike)
        F, K, T, r = self.forward, strike, self.maturity, self.rate

        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))

        if option_type == 'call':
            return np.exp(-r * T) * norm.cdf(d1)
        else:
            return -np.exp(-r * T) * norm.cdf(-d1)

    def vega(self, strike: float) -> float:
        """
        Calculate option vega.

        ν = e^(-rT) · F · √T · n(d1)

        where n is the standard normal PDF.
        """
        sigma = self.implied_volatility(strike)
        F, K, T, r = self.forward, strike, self.maturity, self.rate

        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))

        return np.exp(-r * T) * F * np.sqrt(T) * norm.pdf(d1)


def sabr_implied_vol(
    strike: float,
    forward: float,
    maturity: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0
) -> float:
    """
    Calculate SABR implied volatility using Hagan's formula.

    The full approximation formula:

    σ_SABR(K, T) = α / (FK)^((1-β)/2) · [1 + A·log²(F/K) + B] · z/x(z)

    where:
        A = (1-β)² / 24
        B = ν² / (24·α²·(FK)^(1-β))

        z = (ν/α) · (FK)^((1-β)/2) · log(F/K)
        x(z) = log[(√(1-2ρz+z²) + z - ρ) / (1-ρ)]

    For ATM (F ≈ K):
        σ_ATM = α/F^(1-β) · [1 + ((1-β)²α²/(24F^(2-2β)) + ρβνα/(4F^(1-β)) + (2-3ρ²)ν²/24)·T]

    Args:
        strike: Option strike K
        forward: Forward price F
        maturity: Time to expiry T
        alpha: SABR α parameter
        beta: SABR β parameter
        rho: SABR ρ parameter
        nu: SABR ν parameter
        shift: Shift for shifted SABR (for negative rates)

    Returns:
        SABR implied volatility
    """
    F = forward + shift
    K = strike + shift

    if F <= 0 or K <= 0:
        raise ValueError(f"Forward ({F}) and strike ({K}) must be positive")

    if maturity <= 0:
        return alpha / (F ** (1 - beta))

    if abs(F - K) < 1e-10:
        F_pow = F ** (1 - beta)
        F_pow2 = F ** (2 * (1 - beta))

        term1 = ((1 - beta)**2 / 24) * (alpha**2 / F_pow2)
        term2 = (rho * beta * nu * alpha) / (4 * F_pow)
        term3 = ((2 - 3 * rho**2) / 24) * nu**2

        return (alpha / F_pow) * (1 + (term1 + term2 + term3) * maturity)

    FK = F * K
    FK_pow = FK ** ((1 - beta) / 2)
    log_FK = np.log(F / K)

    z = (nu / alpha) * FK_pow * log_FK

    if abs(z) < 1e-10:
        x_z = 1.0
    else:
        sqrt_term = np.sqrt(1 - 2 * rho * z + z**2)
        x_z_arg = (sqrt_term + z - rho) / (1 - rho)

        if x_z_arg <= 0:
            x_z = 1.0
        else:
            x_z = z / np.log(x_z_arg)

    A = ((1 - beta)**2 / 24) * (log_FK**2)
    B = ((1 - beta)**4 / 1920) * (log_FK**4)
    C = ((1 - beta)**2 / 24) * (alpha**2 / (FK ** (1 - beta)))
    D = (rho * beta * nu * alpha) / (4 * FK_pow)
    E = ((2 - 3 * rho**2) / 24) * nu**2

    numerator = alpha * (1 + (C + D + E) * maturity)
    denominator = FK_pow * (1 + A + B)

    sigma = (numerator / denominator) * x_z

    return max(sigma, 1e-10)


class SABRSurface:
    """
    SABR Volatility Surface across multiple maturities.

    Each maturity has its own SABR parameters, but parameters
    can be interpolated for consistency.
    """

    def __init__(
        self,
        forward_curve: np.ndarray,
        maturities: np.ndarray,
        rate: float = 0.0
    ):
        """
        Initialize SABR surface.

        Args:
            forward_curve: Forward prices for each maturity
            maturities: Array of maturities
            rate: Risk-free rate
        """
        self.forward_curve = forward_curve
        self.maturities = maturities
        self.rate = rate

        self.sabr_models = {}

        for F, T in zip(forward_curve, maturities):
            self.sabr_models[T] = SABRModel(F, T, rate)

    def set_parameters_for_maturity(
        self,
        maturity: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float
    ) -> None:
        """Set SABR parameters for a specific maturity."""
        if maturity not in self.sabr_models:
            raise ValueError(f"Maturity {maturity} not in surface")

        self.sabr_models[maturity].set_parameters(alpha, beta, rho, nu)

    def implied_volatility(self, strike: float, maturity: float) -> float:
        """Get implied volatility at (strike, maturity)."""
        if maturity in self.sabr_models:
            return self.sabr_models[maturity].implied_volatility(strike)
        else:
            T_lower = max([T for T in self.maturities if T <= maturity], default=self.maturities[0])
            T_upper = min([T for T in self.maturities if T >= maturity], default=self.maturities[-1])

            if T_lower == T_upper:
                return self.sabr_models[T_lower].implied_volatility(strike)

            vol_lower = self.sabr_models[T_lower].implied_volatility(strike)
            vol_upper = self.sabr_models[T_upper].implied_volatility(strike)

            weight = (maturity - T_lower) / (T_upper - T_lower)
            return vol_lower + weight * (vol_upper - vol_lower)

    def calibrate_all(
        self,
        strikes_by_maturity: dict,
        vols_by_maturity: dict,
        fix_beta: float = 0.5
    ) -> None:
        """
        Calibrate SABR parameters for all maturities.

        Args:
            strikes_by_maturity: Dict mapping maturity -> strikes array
            vols_by_maturity: Dict mapping maturity -> vols array
            fix_beta: Fixed beta value
        """
        for T in self.maturities:
            if T in strikes_by_maturity and T in vols_by_maturity:
                self.sabr_models[T].calibrate(
                    strikes_by_maturity[T],
                    vols_by_maturity[T],
                    fix_beta=fix_beta
                )

    def get_surface(
        self,
        strikes: np.ndarray
    ) -> np.ndarray:
        """
        Get full implied volatility surface.

        Args:
            strikes: Array of strikes

        Returns:
            2D array of implied vols (strikes x maturities)
        """
        surface = np.zeros((len(strikes), len(self.maturities)))

        for i, K in enumerate(strikes):
            for j, T in enumerate(self.maturities):
                surface[i, j] = self.implied_volatility(K, T)

        return surface
