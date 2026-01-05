"""
Alpaca Options Data Fetcher

Fetches options chain data including greeks and implied volatility
from the Alpaca Markets API.
"""

import os
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import (
    OptionChainRequest,
    OptionLatestQuoteRequest,
    StockLatestQuoteRequest,
)


class AlpacaOptionsFetcher:
    """Fetches options data from Alpaca Markets API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True,
    ):
        """
        Initialize the Alpaca options fetcher.

        Args:
            api_key: Alpaca API key (or set ALPACA_API_KEY env var)
            api_secret: Alpaca API secret (or set ALPACA_API_SECRET env var)
            paper: Use paper trading environment (default True)
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and "
                "ALPACA_API_SECRET environment variables or pass them directly."
            )

        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
            paper=paper,
        )

        self.option_data_client = OptionHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
        )

        self.stock_data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
        )

    def get_underlying_price(self, symbol: str) -> float:
        """
        Get the current price of the underlying asset.

        Args:
            symbol: Stock/ETF symbol (e.g., 'SPY', 'QQQ')

        Returns:
            Current mid-price of the underlying
        """
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = self.stock_data_client.get_stock_latest_quote(request)[symbol]
        return (quote.bid_price + quote.ask_price) / 2

    def get_option_contracts(
        self,
        symbol: str,
        min_expiration_days: int = 7,
        max_expiration_days: int = 90,
        strike_range_pct: float = 0.20,
        contract_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get available option contracts for a symbol.

        Args:
            symbol: Underlying symbol (e.g., 'SPY', 'QQQ')
            min_expiration_days: Minimum days to expiration
            max_expiration_days: Maximum days to expiration
            strike_range_pct: Strike price range as percentage of spot price
            contract_type: 'call', 'put', or None for both

        Returns:
            DataFrame with option contract details
        """
        underlying_price = self.get_underlying_price(symbol)
        min_strike = underlying_price * (1 - strike_range_pct)
        max_strike = underlying_price * (1 + strike_range_pct)

        min_expiry = datetime.now().date() + timedelta(days=min_expiration_days)
        max_expiry = datetime.now().date() + timedelta(days=max_expiration_days)

        contracts_list = []

        # Fetch calls and puts separately if needed
        types_to_fetch = []
        if contract_type is None:
            types_to_fetch = [ContractType.CALL, ContractType.PUT]
        elif contract_type.lower() == "call":
            types_to_fetch = [ContractType.CALL]
        else:
            types_to_fetch = [ContractType.PUT]

        for ctype in types_to_fetch:
            request = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                strike_price_gte=str(min_strike),
                strike_price_lte=str(max_strike),
                expiration_date_gte=min_expiry,
                expiration_date_lte=max_expiry,
                status=AssetStatus.ACTIVE,
                type=ctype,
            )

            response = self.trading_client.get_option_contracts(request)

            if response.option_contracts:
                for contract in response.option_contracts:
                    contracts_list.append(
                        {
                            "symbol": contract.symbol,
                            "underlying": contract.underlying_symbol,
                            "strike": float(contract.strike_price),
                            "expiration": contract.expiration_date,
                            "type": contract.type.value,
                            "style": contract.style.value if contract.style else "american",
                        }
                    )

        df = pd.DataFrame(contracts_list)
        if not df.empty:
            df["expiration"] = pd.to_datetime(df["expiration"])
            df = df.sort_values(["expiration", "strike", "type"])

        return df

    def get_option_chain_with_greeks(
        self,
        symbol: str,
        min_expiration_days: int = 7,
        max_expiration_days: int = 90,
        strike_range_pct: float = 0.20,
    ) -> pd.DataFrame:
        """
        Get full option chain with greeks and implied volatility.

        Args:
            symbol: Underlying symbol (e.g., 'SPY', 'QQQ')
            min_expiration_days: Minimum days to expiration
            max_expiration_days: Maximum days to expiration
            strike_range_pct: Strike price range as percentage of spot price

        Returns:
            DataFrame with options data including IV and greeks
        """
        underlying_price = self.get_underlying_price(symbol)

        # Get option chain using the data API
        request = OptionChainRequest(underlying_symbol=symbol)

        try:
            chain_data = self.option_data_client.get_option_chain(request)
        except Exception as e:
            print(f"Warning: Could not fetch option chain via data API: {e}")
            print("Falling back to contract-by-contract fetching...")
            return self._fetch_chain_fallback(
                symbol,
                min_expiration_days,
                max_expiration_days,
                strike_range_pct,
            )

        options_list = []

        for contract_symbol, snapshot in chain_data.items():
            # Parse contract symbol to extract details
            # Format: SPY250117C00585000
            parsed = self._parse_option_symbol(contract_symbol)
            if parsed is None:
                continue

            strike = parsed["strike"]
            expiration = parsed["expiration"]
            option_type = parsed["type"]

            # Filter by expiration range
            days_to_expiry = (expiration - datetime.now().date()).days
            if days_to_expiry < min_expiration_days or days_to_expiry > max_expiration_days:
                continue

            # Filter by strike range
            if strike < underlying_price * (1 - strike_range_pct):
                continue
            if strike > underlying_price * (1 + strike_range_pct):
                continue

            # Extract data from snapshot
            option_data = {
                "symbol": contract_symbol,
                "underlying": symbol,
                "strike": strike,
                "expiration": expiration,
                "type": option_type,
                "days_to_expiry": days_to_expiry,
            }

            # Latest quote data
            if snapshot.latest_quote:
                option_data["bid"] = snapshot.latest_quote.bid_price
                option_data["ask"] = snapshot.latest_quote.ask_price
                option_data["mid_price"] = (
                    snapshot.latest_quote.bid_price + snapshot.latest_quote.ask_price
                ) / 2

            # Latest trade data
            if snapshot.latest_trade:
                option_data["last_price"] = snapshot.latest_trade.price
                option_data["last_size"] = snapshot.latest_trade.size

            # Greeks data
            if snapshot.greeks:
                option_data["delta"] = snapshot.greeks.delta
                option_data["gamma"] = snapshot.greeks.gamma
                option_data["theta"] = snapshot.greeks.theta
                option_data["vega"] = snapshot.greeks.vega
                option_data["rho"] = snapshot.greeks.rho

            # Implied volatility
            if snapshot.implied_volatility is not None:
                option_data["iv"] = snapshot.implied_volatility

            options_list.append(option_data)

        df = pd.DataFrame(options_list)

        if not df.empty:
            df["expiration"] = pd.to_datetime(df["expiration"])
            df = df.sort_values(["expiration", "strike", "type"])
            df["underlying_price"] = underlying_price
            df["moneyness"] = df["strike"] / underlying_price

        return df

    def _parse_option_symbol(self, symbol: str) -> Optional[dict]:
        """
        Parse an OCC option symbol.

        Format: UNDERLYING + YYMMDD + C/P + STRIKE (8 digits, 3 decimal implied)
        Example: SPY250117C00585000 = SPY Jan 17 2025 $585 Call

        Args:
            symbol: OCC option symbol

        Returns:
            Dictionary with parsed components or None if invalid
        """
        try:
            # Find where the date starts (after underlying symbol)
            # Underlying is variable length, date is YYMMDD
            # Work backwards: last 8 chars are strike, before that is C/P,
            # before that is YYMMDD (6 chars)

            strike_str = symbol[-8:]
            option_type_char = symbol[-9]
            date_str = symbol[-15:-9]
            underlying = symbol[:-15]

            strike = float(strike_str) / 1000

            year = 2000 + int(date_str[0:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            expiration = datetime(year, month, day).date()

            option_type = "call" if option_type_char == "C" else "put"

            return {
                "underlying": underlying,
                "expiration": expiration,
                "type": option_type,
                "strike": strike,
            }
        except (ValueError, IndexError):
            return None

    def _fetch_chain_fallback(
        self,
        symbol: str,
        min_expiration_days: int,
        max_expiration_days: int,
        strike_range_pct: float,
    ) -> pd.DataFrame:
        """
        Fallback method to fetch option chain contract by contract.

        This is slower but works if the bulk option chain endpoint fails.
        """
        contracts_df = self.get_option_contracts(
            symbol,
            min_expiration_days,
            max_expiration_days,
            strike_range_pct,
        )

        if contracts_df.empty:
            return pd.DataFrame()

        underlying_price = self.get_underlying_price(symbol)
        options_list = []

        # Fetch quotes in batches
        contract_symbols = contracts_df["symbol"].tolist()
        batch_size = 100

        for i in range(0, len(contract_symbols), batch_size):
            batch = contract_symbols[i : i + batch_size]

            try:
                request = OptionLatestQuoteRequest(symbol_or_symbols=batch)
                quotes = self.option_data_client.get_option_latest_quote(request)

                for contract_symbol, quote in quotes.items():
                    contract_info = contracts_df[
                        contracts_df["symbol"] == contract_symbol
                    ].iloc[0]

                    days_to_expiry = (
                        contract_info["expiration"] - pd.Timestamp.now()
                    ).days

                    options_list.append(
                        {
                            "symbol": contract_symbol,
                            "underlying": symbol,
                            "strike": contract_info["strike"],
                            "expiration": contract_info["expiration"],
                            "type": contract_info["type"],
                            "days_to_expiry": days_to_expiry,
                            "bid": quote.bid_price,
                            "ask": quote.ask_price,
                            "mid_price": (quote.bid_price + quote.ask_price) / 2,
                        }
                    )
            except Exception as e:
                print(f"Warning: Failed to fetch batch quotes: {e}")
                continue

        df = pd.DataFrame(options_list)

        if not df.empty:
            df["underlying_price"] = underlying_price
            df["moneyness"] = df["strike"] / underlying_price

        return df


def fetch_volatility_surface_data(
    symbol: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    min_expiration_days: int = 7,
    max_expiration_days: int = 90,
    strike_range_pct: float = 0.20,
) -> pd.DataFrame:
    """
    Convenience function to fetch volatility surface data.

    Args:
        symbol: Underlying symbol (e.g., 'SPY', 'QQQ')
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        min_expiration_days: Minimum days to expiration
        max_expiration_days: Maximum days to expiration
        strike_range_pct: Strike price range as percentage of spot

    Returns:
        DataFrame ready for volatility surface construction
    """
    fetcher = AlpacaOptionsFetcher(api_key, api_secret)
    return fetcher.get_option_chain_with_greeks(
        symbol,
        min_expiration_days,
        max_expiration_days,
        strike_range_pct,
    )
