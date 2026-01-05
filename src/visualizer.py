"""
3D Volatility Surface Visualizer

Creates 3D visualizations of implied volatility surfaces
and exports them as video files.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
import imageio
from tqdm import tqdm
import tempfile
import os

from .volatility_surface import VolatilitySurface


@dataclass
class VisualizationConfig:
    """Configuration for volatility surface visualization."""

    # Figure settings
    figsize: Tuple[int, int] = (14, 10)
    dpi: int = 100

    # Color settings
    colormap: str = "RdYlBu_r"  # Red-Yellow-Blue reversed (red = high vol)
    alpha: float = 0.85

    # Axis labels
    x_label: str = "Moneyness (K/S)"
    y_label: str = "Days to Expiration"
    z_label: str = "Implied Volatility (%)"

    # Title settings
    title_fontsize: int = 16
    label_fontsize: int = 12

    # View angles for rotation
    elevation_start: float = 25
    elevation_end: float = 35
    azimuth_start: float = -60
    azimuth_end: float = 300  # Full rotation + some

    # Animation settings
    fps: int = 30
    duration_seconds: float = 10.0
    pause_frames: int = 30  # Pause at start/end

    # Additional features
    show_colorbar: bool = True
    show_contours: bool = True
    contour_levels: int = 10

    # Grid lines
    show_grid: bool = True
    grid_alpha: float = 0.3

    @property
    def total_frames(self) -> int:
        """Total number of frames in the animation."""
        return int(self.fps * self.duration_seconds) + 2 * self.pause_frames


class VolatilitySurfaceVisualizer:
    """Creates 3D visualizations of volatility surfaces."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self._setup_style()

    def _setup_style(self):
        """Set up matplotlib style."""
        plt.style.use("dark_background")

        # Custom color adjustments for better visibility
        plt.rcParams["figure.facecolor"] = "#1a1a2e"
        plt.rcParams["axes.facecolor"] = "#16213e"
        plt.rcParams["axes.edgecolor"] = "#e0e0e0"
        plt.rcParams["axes.labelcolor"] = "#e0e0e0"
        plt.rcParams["text.color"] = "#e0e0e0"
        plt.rcParams["xtick.color"] = "#e0e0e0"
        plt.rcParams["ytick.color"] = "#e0e0e0"
        plt.rcParams["grid.color"] = "#4a4a6a"
        plt.rcParams["grid.alpha"] = 0.3

    def _create_colormap(self) -> LinearSegmentedColormap:
        """Create a custom colormap for volatility visualization."""
        colors = [
            "#1a237e",  # Deep blue (low vol)
            "#1976d2",  # Blue
            "#4caf50",  # Green
            "#ffeb3b",  # Yellow
            "#ff9800",  # Orange
            "#f44336",  # Red (high vol)
            "#b71c1c",  # Dark red (very high vol)
        ]
        return LinearSegmentedColormap.from_list("volatility", colors, N=256)

    def create_static_plot(
        self,
        surface: VolatilitySurface,
        output_path: Optional[str] = None,
        elevation: float = 30,
        azimuth: float = -60,
    ) -> plt.Figure:
        """
        Create a static 3D plot of the volatility surface.

        Args:
            surface: VolatilitySurface to visualize
            output_path: Path to save the figure (optional)
            elevation: Viewing elevation angle
            azimuth: Viewing azimuth angle

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection="3d")

        # Get data
        X = surface.strike_grid
        Y = surface.expiry_grid
        Z = surface.ivs

        # Create custom colormap
        cmap = self._create_colormap()

        # Plot surface
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cmap,
            alpha=self.config.alpha,
            antialiased=True,
            linewidth=0.2,
            edgecolor="white",
            rcount=100,
            ccount=100,
        )

        # Add contour projection on the bottom
        if self.config.show_contours:
            z_offset = Z.min() - (Z.max() - Z.min()) * 0.1
            ax.contourf(
                X,
                Y,
                Z,
                levels=self.config.contour_levels,
                cmap=cmap,
                alpha=0.3,
                offset=z_offset,
            )

        # Set labels
        ax.set_xlabel(self.config.x_label, fontsize=self.config.label_fontsize, labelpad=10)
        ax.set_ylabel(self.config.y_label, fontsize=self.config.label_fontsize, labelpad=10)
        ax.set_zlabel(self.config.z_label, fontsize=self.config.label_fontsize, labelpad=10)

        # Set title
        title = f"{surface.symbol} Implied Volatility Surface\n{surface.timestamp}"
        ax.set_title(title, fontsize=self.config.title_fontsize, pad=20)

        # Set view angle
        ax.view_init(elev=elevation, azim=azimuth)

        # Add colorbar
        if self.config.show_colorbar:
            cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
            cbar.set_label("IV (%)", fontsize=self.config.label_fontsize)

        # Customize grid
        if self.config.show_grid:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("#4a4a6a")
            ax.yaxis.pane.set_edgecolor("#4a4a6a")
            ax.zaxis.pane.set_edgecolor("#4a4a6a")

        # Add underlying price annotation
        ax.text2D(
            0.02,
            0.98,
            f"Underlying: ${surface.underlying_price:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            color="#e0e0e0",
        )

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())

        return fig

    def create_animation_frame(
        self,
        ax: Axes3D,
        surface: VolatilitySurface,
        frame_num: int,
        surf_plot: Optional[object] = None,
    ) -> Tuple[float, float]:
        """
        Set up a single animation frame.

        Args:
            ax: 3D axes object
            surface: VolatilitySurface to visualize
            frame_num: Current frame number
            surf_plot: Existing surface plot (for updates)

        Returns:
            Tuple of (elevation, azimuth) angles
        """
        total_rotation_frames = self.config.total_frames - 2 * self.config.pause_frames

        # Calculate view angles with smooth easing
        if frame_num < self.config.pause_frames:
            # Pause at start
            progress = 0
        elif frame_num >= self.config.total_frames - self.config.pause_frames:
            # Pause at end
            progress = 1
        else:
            # Smooth rotation with easing
            linear_progress = (frame_num - self.config.pause_frames) / total_rotation_frames
            # Apply ease-in-out cubic
            if linear_progress < 0.5:
                progress = 4 * linear_progress**3
            else:
                progress = 1 - (-2 * linear_progress + 2) ** 3 / 2

        elevation = self.config.elevation_start + progress * (
            self.config.elevation_end - self.config.elevation_start
        )
        azimuth = self.config.azimuth_start + progress * (
            self.config.azimuth_end - self.config.azimuth_start
        )

        ax.view_init(elev=elevation, azim=azimuth)

        return elevation, azimuth

    def render_video(
        self,
        surface: VolatilitySurface,
        output_path: str,
        show_progress: bool = True,
    ) -> str:
        """
        Render the volatility surface animation as a video file.

        Args:
            surface: VolatilitySurface to visualize
            output_path: Path for output video file
            show_progress: Show progress bar

        Returns:
            Path to the created video file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure output has video extension
        if output_path.suffix.lower() not in [".mp4", ".avi", ".mov", ".gif"]:
            output_path = output_path.with_suffix(".mp4")

        # Create figure
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection="3d")

        # Get data
        X = surface.strike_grid
        Y = surface.expiry_grid
        Z = surface.ivs

        # Create colormap
        cmap = self._create_colormap()

        # Plot surface
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cmap,
            alpha=self.config.alpha,
            antialiased=True,
            linewidth=0.1,
            edgecolor="white",
            rcount=80,
            ccount=80,
        )

        # Add contour projection
        if self.config.show_contours:
            z_offset = Z.min() - (Z.max() - Z.min()) * 0.1
            ax.contourf(
                X,
                Y,
                Z,
                levels=self.config.contour_levels,
                cmap=cmap,
                alpha=0.3,
                offset=z_offset,
            )

        # Set labels
        ax.set_xlabel(self.config.x_label, fontsize=self.config.label_fontsize, labelpad=10)
        ax.set_ylabel(self.config.y_label, fontsize=self.config.label_fontsize, labelpad=10)
        ax.set_zlabel(self.config.z_label, fontsize=self.config.label_fontsize, labelpad=10)

        # Set title
        title = f"{surface.symbol} Implied Volatility Surface\n{surface.timestamp}"
        ax.set_title(title, fontsize=self.config.title_fontsize, pad=20)

        # Add colorbar
        if self.config.show_colorbar:
            cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
            cbar.set_label("IV (%)", fontsize=self.config.label_fontsize)

        # Customize panes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#4a4a6a")
        ax.yaxis.pane.set_edgecolor("#4a4a6a")
        ax.zaxis.pane.set_edgecolor("#4a4a6a")

        # Add underlying price annotation
        ax.text2D(
            0.02,
            0.98,
            f"Underlying: ${surface.underlying_price:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            color="#e0e0e0",
        )

        plt.tight_layout()

        # Use imageio for video creation
        frames = []
        frame_iterator = range(self.config.total_frames)

        if show_progress:
            frame_iterator = tqdm(
                frame_iterator, desc="Rendering frames", unit="frame"
            )

        for frame_num in frame_iterator:
            elevation, azimuth = self.create_animation_frame(ax, surface, frame_num)

            # Capture frame
            fig.canvas.draw()

            # Convert to numpy array (compatible with matplotlib 3.8+)
            buf = fig.canvas.buffer_rgba()
            data = np.asarray(buf)
            # Convert RGBA to RGB
            data = data[:, :, :3]
            frames.append(data.copy())

        plt.close(fig)

        # Write video
        if show_progress:
            print(f"Writing video to {output_path}...")

        if output_path.suffix.lower() == ".gif":
            imageio.mimsave(
                str(output_path),
                frames,
                fps=self.config.fps,
                loop=0,
            )
        else:
            imageio.mimsave(
                str(output_path),
                frames,
                fps=self.config.fps,
                codec="libx264",
                quality=8,
            )

        if show_progress:
            print(f"Video saved: {output_path}")

        return str(output_path)

    def render_video_matplotlib(
        self,
        surface: VolatilitySurface,
        output_path: str,
        show_progress: bool = True,
    ) -> str:
        """
        Render video using matplotlib's animation module.

        Alternative method that may work better in some environments.

        Args:
            surface: VolatilitySurface to visualize
            output_path: Path for output video file
            show_progress: Show progress bar

        Returns:
            Path to the created video file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() not in [".mp4", ".avi", ".mov"]:
            output_path = output_path.with_suffix(".mp4")

        # Create figure
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection="3d")

        # Get data
        X = surface.strike_grid
        Y = surface.expiry_grid
        Z = surface.ivs

        cmap = self._create_colormap()

        # Initial plot
        surf = ax.plot_surface(
            X, Y, Z, cmap=cmap, alpha=self.config.alpha, antialiased=True
        )

        ax.set_xlabel(self.config.x_label, fontsize=self.config.label_fontsize)
        ax.set_ylabel(self.config.y_label, fontsize=self.config.label_fontsize)
        ax.set_zlabel(self.config.z_label, fontsize=self.config.label_fontsize)

        title = f"{surface.symbol} Implied Volatility Surface\n{surface.timestamp}"
        ax.set_title(title, fontsize=self.config.title_fontsize)

        def update(frame_num):
            self.create_animation_frame(ax, surface, frame_num)
            return []

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=self.config.total_frames,
            interval=1000 / self.config.fps,
            blit=False,
        )

        # Save animation
        writer = animation.FFMpegWriter(
            fps=self.config.fps,
            metadata={"title": f"{surface.symbol} Volatility Surface"},
            bitrate=5000,
        )

        if show_progress:
            print(f"Rendering animation to {output_path}...")

        anim.save(str(output_path), writer=writer)
        plt.close(fig)

        if show_progress:
            print(f"Video saved: {output_path}")

        return str(output_path)


def create_demo_video(output_path: str = "volatility_surface_demo.mp4") -> str:
    """
    Create a demo video using sample volatility surface data.

    Args:
        output_path: Path for the output video

    Returns:
        Path to the created video file
    """
    from .volatility_surface import create_sample_surface_data

    surface = create_sample_surface_data()
    visualizer = VolatilitySurfaceVisualizer()
    return visualizer.render_video(surface, output_path)
