"""
Implied Volatility Surface Calculator

Constructs and interpolates implied volatility surfaces from options data.
Includes Black-Scholes IV calculation for fallback scenarios.
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class VolatilitySurface:
    """Container for volatility surface data."""

    strikes: np.ndarray  # Strike prices or moneyness values
    expirations: np.ndarray  # Days to expiration
    ivs: np.ndarray  # 2D array of implied volatilities
    underlying_price: float
    symbol: str
    timestamp: str
    strike_grid: np.ndarray  # Meshgrid X
    expiry_grid: np.ndarray  # Meshgrid Y

    def get_iv(self, strike: float, days_to_expiry: int) -> float:
        """Interpolate IV at a specific strike and expiration."""
        # Simple bilinear interpolation
        strike_idx = np.searchsorted(self.strikes, strike)
        expiry_idx = np.searchsorted(self.expirations, days_to_expiry)

        strike_idx = np.clip(strike_idx, 1, len(self.strikes) - 1)
        expiry_idx = np.clip(expiry_idx, 1, len(self.expirations) - 1)

        return self.ivs[expiry_idx, strike_idx]


class BlackScholes:
    """Black-Scholes model for option pricing and IV calculation."""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes call option price."""
        if T <= 0:
            return max(S - K, 0)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes put option price."""
        if T <= 0:
            return max(K - S, 0)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def implied_volatility(
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        max_iterations: int = 100,
    ) -> Optional[float]:
        """
        Calculate implied volatility using Brent's method.

        Args:
            price: Market price of the option
            S: Underlying spot price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            option_type: 'call' or 'put'
            max_iterations: Maximum iterations for solver

        Returns:
            Implied volatility or None if calculation fails
        """
        if T <= 0 or price <= 0:
            return None

        # Check for intrinsic value bounds
        if option_type == "call":
            intrinsic = max(S - K * np.exp(-r * T), 0)
            if price < intrinsic:
                return None
            price_func = lambda sigma: BlackScholes.call_price(S, K, T, r, sigma) - price
        else:
            intrinsic = max(K * np.exp(-r * T) - S, 0)
            if price < intrinsic:
                return None
            price_func = lambda sigma: BlackScholes.put_price(S, K, T, r, sigma) - price

        try:
            # Search for IV between 0.01 (1%) and 5.0 (500%)
            iv = brentq(price_func, 0.01, 5.0, maxiter=max_iterations)
            return iv
        except (ValueError, RuntimeError):
            return None


class VolatilitySurfaceBuilder:
    """Builds implied volatility surfaces from options data."""

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize the surface builder.

        Args:
            risk_free_rate: Risk-free interest rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_missing_ivs(
        self, df: pd.DataFrame, underlying_price: float
    ) -> pd.DataFrame:
        """
        Calculate implied volatility for options missing IV data.

        Args:
            df: Options DataFrame with price data
            underlying_price: Current underlying price

        Returns:
            DataFrame with IV column filled in
        """
        df = df.copy()

        if "iv" not in df.columns:
            df["iv"] = np.nan

        missing_iv = df["iv"].isna()

        if not missing_iv.any():
            return df

        for idx in df[missing_iv].index:
            row = df.loc[idx]

            # Get option price (prefer mid, then last)
            if pd.notna(row.get("mid_price")):
                price = row["mid_price"]
            elif pd.notna(row.get("last_price")):
                price = row["last_price"]
            else:
                continue

            # Time to expiry in years
            T = row["days_to_expiry"] / 365.0

            # Calculate IV
            iv = BlackScholes.implied_volatility(
                price=price,
                S=underlying_price,
                K=row["strike"],
                T=T,
                r=self.risk_free_rate,
                option_type=row["type"],
            )

            if iv is not None:
                df.loc[idx, "iv"] = iv

        return df

    def build_surface(
        self,
        df: pd.DataFrame,
        use_moneyness: bool = True,
        interpolation_method: str = "cubic",
        grid_resolution: int = 50,
    ) -> VolatilitySurface:
        """
        Build an implied volatility surface from options data.

        Args:
            df: Options DataFrame with IV data
            use_moneyness: Use moneyness (K/S) instead of absolute strikes
            interpolation_method: Interpolation method ('linear', 'cubic')
            grid_resolution: Number of points in each dimension

        Returns:
            VolatilitySurface object
        """
        # Get underlying price
        if "underlying_price" in df.columns:
            underlying_price = df["underlying_price"].iloc[0]
        else:
            underlying_price = df["strike"].median()

        # Calculate missing IVs
        df = self.calculate_missing_ivs(df, underlying_price)

        # Filter to valid IV data
        valid_data = df[df["iv"].notna() & (df["iv"] > 0) & (df["iv"] < 3)]

        if len(valid_data) < 10:
            raise ValueError(
                f"Insufficient valid IV data points: {len(valid_data)}. "
                "Need at least 10 points to build surface."
            )

        # Prepare coordinates
        if use_moneyness:
            x_values = valid_data["moneyness"].values
            x_label = "moneyness"
        else:
            x_values = valid_data["strike"].values
            x_label = "strike"

        y_values = valid_data["days_to_expiry"].values
        z_values = valid_data["iv"].values

        # Create regular grid for interpolation
        x_min, x_max = x_values.min(), x_values.max()
        y_min, y_max = y_values.min(), y_values.max()

        # Add small padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.05
        x_max += x_range * 0.05
        y_min = max(1, y_min - y_range * 0.05)
        y_max += y_range * 0.05

        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Interpolate surface
        try:
            Z = interpolate.griddata(
                points=(x_values, y_values),
                values=z_values,
                xi=(X, Y),
                method=interpolation_method,
                fill_value=np.nan,
            )
        except Exception:
            # Fall back to linear interpolation
            Z = interpolate.griddata(
                points=(x_values, y_values),
                values=z_values,
                xi=(X, Y),
                method="linear",
                fill_value=np.nan,
            )

        # Fill any remaining NaN values with nearest neighbor
        if np.any(np.isnan(Z)):
            Z_nearest = interpolate.griddata(
                points=(x_values, y_values),
                values=z_values,
                xi=(X, Y),
                method="nearest",
            )
            Z = np.where(np.isnan(Z), Z_nearest, Z)

        # Convert IV to percentage for display
        Z = Z * 100

        symbol = df["underlying"].iloc[0] if "underlying" in df.columns else "UNKNOWN"
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        return VolatilitySurface(
            strikes=x_grid,
            expirations=y_grid,
            ivs=Z,
            underlying_price=underlying_price,
            symbol=symbol,
            timestamp=timestamp,
            strike_grid=X,
            expiry_grid=Y,
        )

    def build_call_put_surfaces(
        self,
        df: pd.DataFrame,
        use_moneyness: bool = True,
        interpolation_method: str = "cubic",
        grid_resolution: int = 50,
    ) -> Tuple[VolatilitySurface, VolatilitySurface]:
        """
        Build separate surfaces for calls and puts.

        Args:
            df: Options DataFrame with IV data
            use_moneyness: Use moneyness instead of absolute strikes
            interpolation_method: Interpolation method
            grid_resolution: Grid resolution

        Returns:
            Tuple of (call_surface, put_surface)
        """
        calls_df = df[df["type"] == "call"]
        puts_df = df[df["type"] == "put"]

        call_surface = self.build_surface(
            calls_df, use_moneyness, interpolation_method, grid_resolution
        )
        put_surface = self.build_surface(
            puts_df, use_moneyness, interpolation_method, grid_resolution
        )

        return call_surface, put_surface

    def build_combined_surface(
        self,
        df: pd.DataFrame,
        use_moneyness: bool = True,
        interpolation_method: str = "cubic",
        grid_resolution: int = 50,
    ) -> VolatilitySurface:
        """
        Build a combined surface using OTM options (more liquid).

        Uses OTM calls for strikes above spot and OTM puts for strikes below.

        Args:
            df: Options DataFrame with IV data
            use_moneyness: Use moneyness instead of absolute strikes
            interpolation_method: Interpolation method
            grid_resolution: Grid resolution

        Returns:
            Combined VolatilitySurface
        """
        df = df.copy()

        if "underlying_price" in df.columns:
            underlying_price = df["underlying_price"].iloc[0]
        else:
            underlying_price = df["strike"].median()

        # Select OTM options (they tend to be more liquid)
        otm_calls = df[(df["type"] == "call") & (df["strike"] >= underlying_price)]
        otm_puts = df[(df["type"] == "put") & (df["strike"] <= underlying_price)]

        combined_df = pd.concat([otm_calls, otm_puts], ignore_index=True)

        if len(combined_df) < 10:
            # Fall back to using all options
            combined_df = df

        return self.build_surface(
            combined_df, use_moneyness, interpolation_method, grid_resolution
        )


def create_sample_surface_data() -> VolatilitySurface:
    """
    Create sample volatility surface data for testing/demo purposes.

    Returns:
        Sample VolatilitySurface with realistic volatility smile
    """
    # Create grid
    moneyness = np.linspace(0.85, 1.15, 50)
    days_to_expiry = np.linspace(7, 90, 50)
    X, Y = np.meshgrid(moneyness, days_to_expiry)

    # Create realistic volatility smile/smirk
    # Base volatility decreases with time (term structure)
    base_vol = 20 + 10 * np.exp(-Y / 30)

    # Volatility smile - higher IV for OTM options
    smile = 5 * (X - 1) ** 2

    # Volatility skew - higher IV for OTM puts (downside protection premium)
    skew = -8 * (X - 1)

    # Combine effects
    Z = base_vol + smile + skew

    # Add some noise for realism
    np.random.seed(42)
    Z += np.random.normal(0, 0.5, Z.shape)

    # Ensure positive values
    Z = np.maximum(Z, 5)

    return VolatilitySurface(
        strikes=moneyness,
        expirations=days_to_expiry,
        ivs=Z,
        underlying_price=500.0,
        symbol="SAMPLE",
        timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        strike_grid=X,
        expiry_grid=Y,
    )
