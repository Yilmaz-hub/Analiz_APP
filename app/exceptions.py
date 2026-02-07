# exceptions.py
"""Custom exceptions for the ProTrader application."""
class ProTraderError(Exception):
    """Base class for all ProTrader exceptions."""
    pass
class DataFetchError(ProTraderError):
    """Raised when there is an error fetching data from an exchange."""
    pass
class AssetLoadError(ProTraderError):
    """Raised when there is an error loading asset data."""
    pass
class PortfolioError(ProTraderError):
    """Raised when there is an error with portfolio management."""
    pass
class PriceDataError(ProTraderError):
    """Raised when there is an error processing price data."""
    pass
class IndicatorError(ProTraderError):
    """Raised when there is an error calculating technical indicators."""
    pass    
class TelegramError(ProTraderError):
    """Raised when there is an error sending a Telegram notification."""
    pass
