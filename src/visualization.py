"""
Volatility Surface Visualization Module

Provides comprehensive visualization tools for volatility surfaces,
including 3D plots, contour maps, smile analysis, and model comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, List, Dict, Any
import warnings


try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class VolatilitySurfaceVisualizer:
    """
    Comprehensive visualization for volatility surfaces.

    Provides both matplotlib (static) and plotly (interactive) visualizations
    for various aspects of volatility surfaces:

    - 3D surface plots
    - Contour plots (heatmaps)
    - Volatility smiles
    - Term structure
    - Model comparison
    - Arbitrage visualization
    - Local volatility surfaces
    """

    def __init__(
        self,
        style: str = 'default',
        figsize: Tuple[int, int] = (12, 8),
        colormap: str = 'viridis'
    ):
        """
        Initialize the visualizer.

        Args:
            style: Matplotlib style ('default', 'dark_background', 'seaborn', etc.)
            figsize: Default figure size
            colormap: Default colormap for surface plots
        """
        self.style = style
        self.figsize = figsize
        self.colormap = colormap

        if style != 'default':
            try:
                plt.style.use(style)
            except:
                warnings.warn(f"Style '{style}' not available, using default")

    def plot_surface_3d(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        title: str = "Implied Volatility Surface",
        xlabel: str = "Strike",
        ylabel: str = "Maturity (years)",
        zlabel: str = "Implied Volatility",
        interactive: bool = False,
        spot: Optional[float] = None,
        ax: Optional[plt.Axes] = None
    ) -> Any:
        """
        Create a 3D surface plot of the volatility surface.

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            implied_vols: 2D array of implied volatilities
            title: Plot title
            xlabel, ylabel, zlabel: Axis labels
            interactive: Use plotly for interactive plot
            spot: Spot price for moneyness reference line
            ax: Existing matplotlib axes

        Returns:
            Figure object (matplotlib or plotly)
        """
        K, T = np.meshgrid(strikes, maturities, indexing='ij')

        if interactive and PLOTLY_AVAILABLE:
            return self._plot_surface_3d_plotly(
                K, T, implied_vols, title, xlabel, ylabel, zlabel, spot
            )
        else:
            return self._plot_surface_3d_matplotlib(
                K, T, implied_vols, title, xlabel, ylabel, zlabel, spot, ax
            )

    def _plot_surface_3d_matplotlib(
        self,
        K: np.ndarray,
        T: np.ndarray,
        vols: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        zlabel: str,
        spot: Optional[float],
        ax: Optional[plt.Axes]
    ) -> plt.Figure:
        """Matplotlib 3D surface plot."""
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        surf = ax.plot_surface(
            K, T, vols * 100,
            cmap=self.colormap,
            alpha=0.8,
            edgecolor='none'
        )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_zlabel(f"{zlabel} (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        if spot is not None:
            T_line = np.linspace(T.min(), T.max(), 50)
            K_line = np.full_like(T_line, spot)
            vol_line = np.interp(spot, K[:, 0], vols[:, 0])
            ax.plot(K_line, T_line, np.full_like(T_line, vol_line * 100),
                   'r--', linewidth=2, label='ATM')

        fig.colorbar(surf, shrink=0.5, aspect=10, label='IV (%)')

        ax.view_init(elev=25, azim=45)

        plt.tight_layout()
        return fig

    def _plot_surface_3d_plotly(
        self,
        K: np.ndarray,
        T: np.ndarray,
        vols: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        zlabel: str,
        spot: Optional[float]
    ) -> go.Figure:
        """Plotly interactive 3D surface plot."""
        fig = go.Figure(data=[go.Surface(
            x=K,
            y=T,
            z=vols * 100,
            colorscale='Viridis',
            colorbar=dict(title='IV (%)')
        )])

        if spot is not None:
            T_line = np.linspace(T.min(), T.max(), 50)
            K_line = np.full_like(T_line, spot)
            vol_line = np.interp(spot, K[:, 0], vols[:, 0])
            fig.add_trace(go.Scatter3d(
                x=K_line,
                y=T_line,
                z=np.full_like(T_line, vol_line * 100),
                mode='lines',
                line=dict(color='red', width=5),
                name='ATM'
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                zaxis_title=f"{zlabel} (%)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
            ),
            width=900,
            height=700
        )

        return fig

    def plot_contour(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        title: str = "Volatility Surface Contour",
        levels: int = 20,
        show_colorbar: bool = True,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Create a contour (heatmap) plot of the volatility surface.

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            implied_vols: 2D array of implied volatilities
            title: Plot title
            levels: Number of contour levels
            show_colorbar: Whether to show colorbar
            ax: Existing axes

        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        K, T = np.meshgrid(strikes, maturities, indexing='ij')

        contour = ax.contourf(
            K, T, implied_vols * 100,
            levels=levels,
            cmap=self.colormap
        )

        ax.contour(K, T, implied_vols * 100, levels=levels, colors='white', alpha=0.3, linewidths=0.5)

        ax.set_xlabel('Strike', fontsize=12)
        ax.set_ylabel('Maturity (years)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        if show_colorbar:
            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label('Implied Volatility (%)', fontsize=10)

        plt.tight_layout()
        return fig

    def plot_smile(
        self,
        strikes: np.ndarray,
        implied_vols: np.ndarray,
        maturity: float,
        spot: Optional[float] = None,
        forward: Optional[float] = None,
        title: Optional[str] = None,
        show_moneyness: bool = False,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Plot volatility smile for a single maturity.

        Args:
            strikes: Array of strikes
            implied_vols: Array of implied volatilities
            maturity: Time to maturity
            spot: Spot price (for ATM line)
            forward: Forward price (for moneyness)
            title: Plot title
            show_moneyness: Plot vs moneyness instead of strike
            ax: Existing axes

        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        if show_moneyness and forward is not None:
            x_values = np.log(strikes / forward)
            xlabel = 'Log-Moneyness (ln(K/F))'
        else:
            x_values = strikes
            xlabel = 'Strike'

        ax.plot(x_values, implied_vols * 100, 'b-', linewidth=2, marker='o', markersize=4)

        if spot is not None and not show_moneyness:
            ax.axvline(x=spot, color='r', linestyle='--', alpha=0.7, label='Spot')
        if forward is not None and not show_moneyness:
            ax.axvline(x=forward, color='g', linestyle='--', alpha=0.7, label='Forward')
        if show_moneyness:
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='ATM')

        if title is None:
            title = f'Volatility Smile (T = {maturity:.3f}y)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Implied Volatility (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_multiple_smiles(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        title: str = "Volatility Smiles Across Maturities",
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Plot multiple volatility smiles on the same axes.

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            implied_vols: 2D array of implied volatilities
            title: Plot title
            ax: Existing axes

        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))
        else:
            fig = ax.figure

        colors = cm.viridis(np.linspace(0.2, 0.8, len(maturities)))

        for j, T in enumerate(maturities):
            ax.plot(
                strikes,
                implied_vols[:, j] * 100,
                color=colors[j],
                linewidth=2,
                label=f'T = {T:.2f}y',
                marker='o',
                markersize=3
            )

        ax.set_xlabel('Strike', fontsize=12)
        ax.set_ylabel('Implied Volatility (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_term_structure(
        self,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        strikes: Optional[np.ndarray] = None,
        strike_labels: Optional[List[str]] = None,
        title: str = "Volatility Term Structure",
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Plot volatility term structure for different strikes.

        Args:
            maturities: Array of maturities
            implied_vols: 2D array (strikes x maturities) or 1D (single strike)
            strikes: Array of strikes (for multiple curves)
            strike_labels: Labels for each strike
            title: Plot title
            ax: Existing axes

        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        if implied_vols.ndim == 1:
            ax.plot(maturities, implied_vols * 100, 'b-', linewidth=2, marker='o')
        else:
            colors = cm.coolwarm(np.linspace(0, 1, len(strikes)))
            for i, K in enumerate(strikes):
                label = strike_labels[i] if strike_labels else f'K = {K:.0f}'
                ax.plot(
                    maturities,
                    implied_vols[i, :] * 100,
                    color=colors[i],
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    label=label
                )
            ax.legend()

        ax.set_xlabel('Maturity (years)', fontsize=12)
        ax.set_ylabel('Implied Volatility (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_local_vs_implied(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        local_vols: np.ndarray,
        maturity_idx: int = 0,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Compare local and implied volatility smiles.

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            implied_vols: Implied vol surface
            local_vols: Local vol surface
            maturity_idx: Index of maturity to plot
            title: Plot title
            ax: Existing axes

        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        T = maturities[maturity_idx]

        ax.plot(strikes, implied_vols[:, maturity_idx] * 100, 'b-',
               linewidth=2, marker='o', label='Implied Vol')
        ax.plot(strikes, local_vols[:, maturity_idx] * 100, 'r--',
               linewidth=2, marker='s', label='Local Vol')

        if title is None:
            title = f'Local vs Implied Volatility (T = {T:.3f}y)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Strike', fontsize=12)
        ax.set_ylabel('Volatility (%)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_model_comparison(
        self,
        strikes: np.ndarray,
        maturity: float,
        market_vols: np.ndarray,
        model_vols: Dict[str, np.ndarray],
        title: str = "Model Comparison",
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Compare different model implied volatilities against market.

        Args:
            strikes: Array of strikes
            maturity: Time to maturity
            market_vols: Market implied volatilities
            model_vols: Dict mapping model names to their implied vols
            title: Plot title
            ax: Existing axes

        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))
        else:
            fig = ax.figure

        ax.plot(strikes, market_vols * 100, 'ko', markersize=8, label='Market')

        colors = ['blue', 'red', 'green', 'orange', 'purple']
        linestyles = ['-', '--', '-.', ':']

        for i, (name, vols) in enumerate(model_vols.items()):
            ax.plot(
                strikes,
                vols * 100,
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2,
                label=name
            )

        ax.set_xlabel('Strike', fontsize=12)
        ax.set_ylabel('Implied Volatility (%)', fontsize=12)
        ax.set_title(f'{title} (T = {maturity:.3f}y)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_arbitrage_check(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        violations: List[Any],
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Visualize arbitrage violations on the surface.

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            implied_vols: 2D array of implied volatilities
            violations: List of ArbitrageViolation objects
            ax: Existing axes

        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = ax.figure

        K, T = np.meshgrid(strikes, maturities, indexing='ij')

        contour = ax.contourf(K, T, implied_vols * 100, levels=20, cmap='Blues', alpha=0.7)

        calendar_violations = [v for v in violations if v.violation_type == 'calendar']
        butterfly_violations = [v for v in violations if v.violation_type == 'butterfly']

        if calendar_violations:
            cal_K = [v.location[0] for v in calendar_violations]
            cal_T = [v.location[1] for v in calendar_violations]
            ax.scatter(cal_K, cal_T, c='red', s=100, marker='x',
                      linewidths=2, label='Calendar Arbitrage')

        if butterfly_violations:
            but_K = [v.location[0] for v in butterfly_violations]
            but_T = [v.location[1] for v in butterfly_violations]
            ax.scatter(but_K, but_T, c='orange', s=100, marker='^',
                      linewidths=2, label='Butterfly Arbitrage')

        ax.set_xlabel('Strike', fontsize=12)
        ax.set_ylabel('Maturity (years)', fontsize=12)
        ax.set_title('Arbitrage Violation Map', fontsize=14, fontweight='bold')
        ax.legend()
        fig.colorbar(contour, ax=ax, label='IV (%)')

        plt.tight_layout()
        return fig

    def plot_heston_paths(
        self,
        time_grid: np.ndarray,
        price_paths: np.ndarray,
        variance_paths: np.ndarray,
        n_display: int = 50,
        title: str = "Heston Model Simulation"
    ) -> plt.Figure:
        """
        Plot simulated Heston model paths.

        Args:
            time_grid: Array of time points
            price_paths: 2D array (time_steps x paths)
            variance_paths: 2D array (time_steps x paths)
            n_display: Number of paths to display
            title: Plot title

        Returns:
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        paths_to_plot = min(n_display, price_paths.shape[1])

        for i in range(paths_to_plot):
            ax1.plot(time_grid, price_paths[:, i], alpha=0.5, linewidth=0.5)

        ax1.plot(time_grid, np.mean(price_paths, axis=1), 'r-', linewidth=2, label='Mean')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.set_title(f'{title} - Price Paths', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for i in range(paths_to_plot):
            ax2.plot(time_grid, np.sqrt(variance_paths[:, i]), alpha=0.5, linewidth=0.5)

        ax2.plot(time_grid, np.mean(np.sqrt(variance_paths), axis=1), 'r-', linewidth=2, label='Mean')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Volatility', fontsize=12)
        ax2.set_title('Variance Paths (√v)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_sabr_parameters(
        self,
        maturities: np.ndarray,
        alphas: np.ndarray,
        rhos: np.ndarray,
        nus: np.ndarray,
        beta: float,
        title: str = "SABR Parameters Across Maturities"
    ) -> plt.Figure:
        """
        Plot SABR parameters across different maturities.

        Args:
            maturities: Array of maturities
            alphas: Alpha values
            rhos: Rho values
            nus: Nu values
            beta: Beta value (constant)
            title: Plot title

        Returns:
            Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].plot(maturities, alphas, 'b-o', linewidth=2)
        axes[0, 0].set_xlabel('Maturity')
        axes[0, 0].set_ylabel('α (Alpha)')
        axes[0, 0].set_title('Volatility Level')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].axhline(y=beta, color='b', linewidth=2)
        axes[0, 1].set_xlabel('Maturity')
        axes[0, 1].set_ylabel('β (Beta)')
        axes[0, 1].set_title(f'CEV Exponent (β = {beta})')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(maturities, rhos, 'g-o', linewidth=2)
        axes[1, 0].set_xlabel('Maturity')
        axes[1, 0].set_ylabel('ρ (Rho)')
        axes[1, 0].set_title('Correlation')
        axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(maturities, nus, 'r-o', linewidth=2)
        axes[1, 1].set_xlabel('Maturity')
        axes[1, 1].set_ylabel('ν (Nu)')
        axes[1, 1].set_title('Vol of Vol')
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def create_dashboard(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        spot: float,
        title: str = "Volatility Surface Dashboard"
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple views.

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            implied_vols: 2D array of implied volatilities
            spot: Current spot price
            title: Dashboard title

        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(16, 12))

        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        self.plot_surface_3d(strikes, maturities, implied_vols, spot=spot, ax=ax1)
        ax1.set_title('3D Surface')

        ax2 = fig.add_subplot(2, 2, 2)
        self.plot_contour(strikes, maturities, implied_vols, ax=ax2)
        ax2.set_title('Contour Map')

        ax3 = fig.add_subplot(2, 2, 3)
        self.plot_multiple_smiles(strikes, maturities, implied_vols, ax=ax3)
        ax3.set_title('Volatility Smiles')

        ax4 = fig.add_subplot(2, 2, 4)
        atm_idx = np.argmin(np.abs(strikes - spot))
        atm_vols = implied_vols[atm_idx, :]
        ax4.plot(maturities, atm_vols * 100, 'b-o', linewidth=2)
        ax4.set_xlabel('Maturity')
        ax4.set_ylabel('ATM IV (%)')
        ax4.set_title('ATM Term Structure')
        ax4.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        return fig

    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        dpi: int = 300,
        format: str = 'png'
    ) -> None:
        """
        Save figure to file.

        Args:
            fig: Figure to save
            filename: Output filename
            dpi: Resolution
            format: File format
        """
        fig.savefig(filename, dpi=dpi, format=format, bbox_inches='tight')
