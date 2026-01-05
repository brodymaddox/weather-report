# Volatility Surface 3D Visualization

A Python tool to visualize implied volatility surfaces from options market data using the Alpaca Markets API. Creates stunning 3D rotating video visualizations of the "volatility smile" that shows expected market movement derived from options prices.

## What This Shows

The **implied volatility surface** is a 3D representation of how the market expects future price movements based on options prices:

- **X-axis (Moneyness)**: Strike price relative to current price (K/S). Values < 1 are in-the-money puts, values > 1 are out-of-the-money calls
- **Y-axis (Days to Expiration)**: How far out the option expires
- **Z-axis (Implied Volatility)**: The market's expectation of future volatility, derived from option prices using Black-Scholes

Key features visible in the surface:
- **Volatility Smile**: Higher IV for deep OTM and ITM options
- **Volatility Skew**: Typically higher IV for puts (downside protection premium)
- **Term Structure**: How volatility expectations change with time

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd weather-report

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### Alpaca API Credentials

1. Sign up at [Alpaca Markets](https://alpaca.markets/)
2. Get your API keys from the [dashboard](https://app.alpaca.markets/)
3. Set environment variables:

```bash
export ALPACA_API_KEY='your-api-key'
export ALPACA_API_SECRET='your-api-secret'
```

Or copy `.env.example` to `.env` and fill in your credentials.

**Note**: Paper trading API works for testing. The Basic (free) plan includes indicative options data. For real-time data, consider Algo Trader Plus subscription.

## Usage

### Basic Usage

```bash
# Generate volatility surface video for SPY
python volatility_viz.py SPY --output spy_volatility.mp4

# Generate for QQQ
python volatility_viz.py QQQ --output qqq_volatility.mp4

# Generate for IWM with custom parameters
python volatility_viz.py IWM --min-days 14 --max-days 60 --output iwm_vol.mp4
```

### Demo Mode (No API Required)

```bash
# Generate demo video with sample data
python volatility_viz.py --demo --output demo_volatility.mp4
```

### Static Image

```bash
# Generate a static PNG image instead of video
python volatility_viz.py SPY --static --output spy_surface.png
```

### All Options

```bash
python volatility_viz.py --help

Options:
  symbol              Underlying symbol (SPY, QQQ, IWM, etc.)
  -o, --output        Output file path
  --min-days          Minimum days to expiration (default: 7)
  --max-days          Maximum days to expiration (default: 90)
  --strike-range      Strike range as decimal (default: 0.20 = 20%)
  --demo              Use demo data (no API required)
  --static            Generate static image instead of video
  --duration          Video duration in seconds (default: 10)
  --fps               Frames per second (default: 30)
  --dpi               Resolution (default: 100)
  --no-contours       Disable contour projection
  --absolute-strikes  Use absolute strike prices instead of moneyness
  --calls-only        Only include call options
  --puts-only         Only include put options
  -v, --verbose       Verbose output
```

## API Reference

### AlpacaOptionsFetcher

```python
from src.alpaca_fetcher import AlpacaOptionsFetcher

fetcher = AlpacaOptionsFetcher()

# Get current price
price = fetcher.get_underlying_price("SPY")

# Get full option chain with Greeks
options_df = fetcher.get_option_chain_with_greeks(
    symbol="SPY",
    min_expiration_days=7,
    max_expiration_days=90,
    strike_range_pct=0.20,
)
```

### VolatilitySurfaceBuilder

```python
from src.volatility_surface import VolatilitySurfaceBuilder

builder = VolatilitySurfaceBuilder(risk_free_rate=0.05)

# Build surface from options data
surface = builder.build_combined_surface(
    df=options_df,
    use_moneyness=True,
    interpolation_method="cubic",
    grid_resolution=50,
)
```

### VolatilitySurfaceVisualizer

```python
from src.visualizer import VolatilitySurfaceVisualizer, VisualizationConfig

config = VisualizationConfig(
    fps=30,
    duration_seconds=10.0,
    show_contours=True,
)

visualizer = VolatilitySurfaceVisualizer(config)

# Render video
visualizer.render_video(surface, "output.mp4")

# Or create static plot
visualizer.create_static_plot(surface, "output.png")
```

## How It Works

1. **Data Fetching**: Uses Alpaca's Options API to get:
   - Option contracts for the specified underlying
   - Latest quotes (bid/ask prices)
   - Greeks (delta, gamma, theta, vega, rho)
   - Implied volatility

2. **Surface Construction**:
   - Calculates missing IVs using Black-Scholes with Brent's method
   - Uses OTM options (more liquid) for combined surface
   - Interpolates using cubic splines on a regular grid

3. **Visualization**:
   - Creates 3D surface plot with custom colormap
   - Adds contour projections for depth perception
   - Renders rotating animation frame-by-frame
   - Exports as MP4 video using imageio

## Alpaca API Notes

Based on [Alpaca's Options Trading Documentation](https://docs.alpaca.markets/docs/options-trading):

- **Option Chain Endpoint**: Provides latest trade, quote, and Greeks for each contract
- **Penny Pilot**: SPY, QQQ, and IWM support penny increments
- **Paper Trading**: Options enabled by default in paper environment
- **Data Plans**: Basic plan includes indicative feed; Algo Trader Plus includes full market data

## Requirements

- Python 3.8+
- alpaca-py >= 0.21.0
- numpy, pandas, scipy
- matplotlib >= 3.7.0
- imageio, imageio-ffmpeg
- tqdm

## License

MIT License - See LICENSE file
