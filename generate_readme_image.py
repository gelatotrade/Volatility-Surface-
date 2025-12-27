"""
Generate a volatility surface visualization for the README.

Creates a professional 3D visualization similar to industry-standard
volatility surface plots with moneyness and term structure axes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


def generate_volatility_surface():
    """
    Generate synthetic volatility surface data with realistic characteristics:
    - Volatility smile (higher vol for OTM options)
    - Negative skew (puts more expensive than calls)
    - Term structure (vol increases then flattens with maturity)
    """
    # Moneyness: 80% to 120% (in percentage)
    moneyness = np.linspace(80, 120, 50)

    # Term: 1 month to 2 years
    term = np.linspace(0.083, 2.0, 40)

    M, T = np.meshgrid(moneyness, term, indexing='ij')

    # Base ATM volatility
    base_vol = 20.0  # 20%

    # Skew: lower moneyness = higher vol (negative skew)
    skew_effect = -0.15 * (M - 100)

    # Smile: OTM options have higher vol (quadratic in moneyness)
    smile_effect = 0.003 * (M - 100)**2

    # Term structure: vol increases with sqrt(T) then flattens
    term_effect = 3.0 * np.sqrt(T) - 0.5 * T

    # Combine effects
    implied_vol = base_vol + skew_effect + smile_effect + term_effect

    # Add slight variation for realism
    np.random.seed(42)
    implied_vol += np.random.normal(0, 0.3, implied_vol.shape)

    # Ensure positive volatilities
    implied_vol = np.maximum(implied_vol, 5.0)

    return moneyness, term, implied_vol


def create_volatility_surface_plot():
    """Create a professional 3D volatility surface visualization."""

    # Generate data
    moneyness, term, implied_vol = generate_volatility_surface()
    M, T = np.meshgrid(moneyness, term, indexing='ij')

    # Create figure with dark background for professional look
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Create custom colormap (yellow -> green -> blue, similar to image)
    colors = ['#FFD700', '#32CD32', '#00CED1', '#1E90FF']
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('vol_surface', colors, N=n_bins)

    # Normalize colors based on volatility values
    norm = mcolors.Normalize(vmin=implied_vol.min(), vmax=implied_vol.max())

    # Plot the surface
    surf = ax.plot_surface(
        M, T, implied_vol,
        cmap=cmap,
        norm=norm,
        alpha=0.95,
        edgecolor='none',
        antialiased=True,
        shade=True
    )

    # Add wireframe for better depth perception
    ax.plot_wireframe(
        M, T, implied_vol,
        color='black',
        alpha=0.1,
        linewidth=0.3,
        rstride=5,
        cstride=5
    )

    # Set labels with professional formatting
    ax.set_xlabel('Moneyness (%)', fontsize=12, labelpad=10)
    ax.set_ylabel('Term (years)', fontsize=12, labelpad=10)
    ax.set_zlabel('Implied Volatility (%)', fontsize=12, labelpad=10)

    # Set title
    ax.set_title('Implied Volatility Surface', fontsize=16, fontweight='bold', pad=20)

    # Adjust view angle for best presentation
    ax.view_init(elev=25, azim=45)

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)
    cbar.set_label('IV (%)', fontsize=11)

    # Set axis limits for cleaner look
    ax.set_xlim(80, 120)
    ax.set_ylim(0, 2.0)

    # Improve tick formatting
    ax.tick_params(labelsize=10)

    # Add grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')

    plt.tight_layout()

    return fig


def create_dual_surface_plot():
    """Create a side-by-side visualization with 3D surface and contour."""

    # Generate data
    moneyness, term, implied_vol = generate_volatility_surface()
    M, T = np.meshgrid(moneyness, term, indexing='ij')

    # Create figure
    fig = plt.figure(figsize=(16, 7))

    # Custom colormap
    colors = ['#FFD700', '#32CD32', '#00CED1', '#1E90FF']
    cmap = mcolors.LinearSegmentedColormap.from_list('vol_surface', colors, N=100)

    # Left plot: 3D surface
    ax1 = fig.add_subplot(121, projection='3d')

    surf = ax1.plot_surface(
        M, T, implied_vol,
        cmap=cmap,
        alpha=0.95,
        edgecolor='none',
        antialiased=True
    )

    ax1.set_xlabel('Moneyness (%)', fontsize=11, labelpad=8)
    ax1.set_ylabel('Term (years)', fontsize=11, labelpad=8)
    ax1.set_zlabel('IV (%)', fontsize=11, labelpad=8)
    ax1.set_title('3D Volatility Surface', fontsize=14, fontweight='bold')
    ax1.view_init(elev=25, azim=45)

    # Right plot: Contour/Heatmap
    ax2 = fig.add_subplot(122)

    contour = ax2.contourf(M, T, implied_vol, levels=30, cmap=cmap)
    ax2.contour(M, T, implied_vol, levels=15, colors='white', alpha=0.3, linewidths=0.5)

    # Add ATM line
    ax2.axvline(x=100, color='red', linestyle='--', alpha=0.7, linewidth=2, label='ATM')

    ax2.set_xlabel('Moneyness (%)', fontsize=11)
    ax2.set_ylabel('Term (years)', fontsize=11)
    ax2.set_title('Volatility Surface Contour', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')

    cbar = fig.colorbar(contour, ax=ax2)
    cbar.set_label('IV (%)', fontsize=11)

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    # Generate single 3D surface plot for README
    print("Generating volatility surface visualization...")

    fig1 = create_volatility_surface_plot()
    fig1.savefig('docs/volatility_surface_example.png', dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("Saved: docs/volatility_surface_example.png")

    # Generate dual plot
    fig2 = create_dual_surface_plot()
    fig2.savefig('docs/volatility_surface_dual.png', dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("Saved: docs/volatility_surface_dual.png")

    plt.close('all')
    print("Done!")
