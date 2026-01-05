"""
Volatility Surface 3D Visualization - Main Entry Point

Fetches options data from Alpaca Markets API and creates
3D visualizations of implied volatility surfaces.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from .alpaca_fetcher import AlpacaOptionsFetcher, fetch_volatility_surface_data
from .volatility_surface import VolatilitySurfaceBuilder, create_sample_surface_data
from .visualizer import VolatilitySurfaceVisualizer, VisualizationConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate 3D implied volatility surface visualizations from options data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate volatility surface video for SPY
  python -m src.main SPY --output spy_volatility.mp4

  # Generate for QQQ with custom expiration range
  python -m src.main QQQ --min-days 14 --max-days 60 --output qqq_vol.mp4

  # Generate demo video without API credentials
  python -m src.main --demo --output demo_volatility.mp4

  # Create static image instead of video
  python -m src.main SPY --static --output spy_surface.png

Environment Variables:
  ALPACA_API_KEY     - Your Alpaca API key
  ALPACA_API_SECRET  - Your Alpaca API secret
        """,
    )

    parser.add_argument(
        "symbol",
        nargs="?",
        default="SPY",
        help="Underlying symbol (e.g., SPY, QQQ, IWM). Default: SPY",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path. Default: {symbol}_volatility_surface.mp4",
    )

    parser.add_argument(
        "--min-days",
        type=int,
        default=7,
        help="Minimum days to expiration. Default: 7",
    )

    parser.add_argument(
        "--max-days",
        type=int,
        default=90,
        help="Maximum days to expiration. Default: 90",
    )

    parser.add_argument(
        "--strike-range",
        type=float,
        default=0.20,
        help="Strike range as decimal (0.20 = 20%% above/below spot). Default: 0.20",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use demo data (no API credentials required)",
    )

    parser.add_argument(
        "--static",
        action="store_true",
        help="Generate static image instead of video",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Video duration in seconds. Default: 10.0",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for video. Default: 30",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Resolution (DPI) of output. Default: 100",
    )

    parser.add_argument(
        "--no-contours",
        action="store_true",
        help="Disable contour projection on surface",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Alpaca API key (or use ALPACA_API_KEY env var)",
    )

    parser.add_argument(
        "--api-secret",
        type=str,
        default=None,
        help="Alpaca API secret (or use ALPACA_API_SECRET env var)",
    )

    parser.add_argument(
        "--absolute-strikes",
        action="store_true",
        help="Use absolute strike prices instead of moneyness",
    )

    parser.add_argument(
        "--calls-only",
        action="store_true",
        help="Only include call options",
    )

    parser.add_argument(
        "--puts-only",
        action="store_true",
        help="Only include put options",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        ext = ".png" if args.static else ".mp4"
        output_path = f"{args.symbol.upper()}_volatility_surface{ext}"

    print(f"=" * 60)
    print(f"  Volatility Surface 3D Visualization")
    print(f"=" * 60)

    # Create visualization config
    config = VisualizationConfig(
        fps=args.fps,
        duration_seconds=args.duration,
        dpi=args.dpi,
        show_contours=not args.no_contours,
    )

    if not args.absolute_strikes:
        config.x_label = "Moneyness (K/S)"
    else:
        config.x_label = "Strike Price ($)"

    visualizer = VolatilitySurfaceVisualizer(config)

    if args.demo:
        print("\n[Demo Mode] Using sample volatility surface data...")
        surface = create_sample_surface_data()
    else:
        print(f"\n[Live Data] Fetching options data for {args.symbol.upper()}...")
        print(f"  Expiration range: {args.min_days} - {args.max_days} days")
        print(f"  Strike range: +/- {args.strike_range * 100:.0f}% from spot")

        try:
            # Fetch options data
            fetcher = AlpacaOptionsFetcher(
                api_key=args.api_key,
                api_secret=args.api_secret,
            )

            print(f"\n  Fetching underlying price...")
            underlying_price = fetcher.get_underlying_price(args.symbol.upper())
            print(f"  {args.symbol.upper()} current price: ${underlying_price:.2f}")

            print(f"\n  Fetching option chain with greeks...")
            options_df = fetcher.get_option_chain_with_greeks(
                symbol=args.symbol.upper(),
                min_expiration_days=args.min_days,
                max_expiration_days=args.max_days,
                strike_range_pct=args.strike_range,
            )

            if options_df.empty:
                print("Error: No options data returned. Check your API credentials and symbol.")
                sys.exit(1)

            # Filter by option type if requested
            if args.calls_only:
                options_df = options_df[options_df["type"] == "call"]
                print(f"  Filtered to calls only")
            elif args.puts_only:
                options_df = options_df[options_df["type"] == "put"]
                print(f"  Filtered to puts only")

            print(f"  Retrieved {len(options_df)} option contracts")

            if args.verbose:
                print(f"\n  Expiration dates: {sorted(options_df['expiration'].unique())}")
                print(f"  Strike range: ${options_df['strike'].min():.2f} - ${options_df['strike'].max():.2f}")

            # Build volatility surface
            print(f"\n  Building volatility surface...")
            builder = VolatilitySurfaceBuilder()

            surface = builder.build_combined_surface(
                df=options_df,
                use_moneyness=not args.absolute_strikes,
                interpolation_method="cubic",
                grid_resolution=50,
            )

            print(f"  Surface built successfully")
            print(f"  IV range: {surface.ivs.min():.1f}% - {surface.ivs.max():.1f}%")

        except ValueError as e:
            print(f"\nError: {e}")
            print("\nTo use live data, set your Alpaca credentials:")
            print("  export ALPACA_API_KEY='your-api-key'")
            print("  export ALPACA_API_SECRET='your-api-secret'")
            print("\nOr run with --demo to use sample data.")
            sys.exit(1)
        except Exception as e:
            print(f"\nError fetching data: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    # Generate visualization
    print(f"\n  Generating {'static image' if args.static else 'video'}...")

    try:
        if args.static:
            fig = visualizer.create_static_plot(surface, output_path)
            print(f"\n  Static image saved: {output_path}")
        else:
            output_file = visualizer.render_video(
                surface,
                output_path,
                show_progress=True,
            )
            print(f"\n  Video saved: {output_file}")

    except Exception as e:
        print(f"\nError generating visualization: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Complete!")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
