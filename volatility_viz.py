#!/usr/bin/env python3
"""
Convenience script to run the volatility surface visualization.

Usage:
    python volatility_viz.py SPY --output spy_volatility.mp4
    python volatility_viz.py QQQ --demo
    python volatility_viz.py --help
"""

from src.main import main

if __name__ == "__main__":
    main()
