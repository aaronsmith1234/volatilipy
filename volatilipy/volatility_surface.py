"""Volatility Surface Class - extension of BlackVarianceSurface object from QuantLib"""


from datetime import datetime
import pandas as pd
import QuantLib as ql

from .helper_functions import (
    _calculate_vols,
    _create_ql_date,
    _create_ql_vol_grid,
    _extract_surface_data,
)
from .options_data import OptionsData


class VolatilitySurface(ql.BlackVarianceSurface):
    """Object to take a dataframe of implied volatilities, with expiry dates along the row
    indices, and strike prices across the column indices, and turn it into a BlackVarianceSurface
    object, which can then be used with Quantlib in order to price options.

    Args:
        options_data (OptionsData): Options data on which to fit the surface
        day_count (ql.DayCounter, optional): Day counter to use with Quantlib.
            Defaults to ql.ActualActual(ql.ActualActual.ISDA).
        calendar (ql.Calendar, optional): Calendar to use with Quantlib.
            Defaults to ql.UnitedStates().
        allow_extrapolation (boolean, optional): Flag to enable extrapolation.
            Defaults to False, consistent with Quantlib.
        interpolation_method (str, optional): Define interpolation method to use.
            Options are "Bilinear", "Bicubic" (and others which I have not explored).
            Defaults to "Bilinear", consistent with Quantlib

    Returns:
        ql.BlackVarianceSurface: Surface of vols, can then be used to price options!
    """

    def __init__(
        self,
        options_data: OptionsData = None,
        volatility_grid: pd.DataFrame = None,
        day_count: ql.DayCounter = ql.ActualActual(ql.ActualActual.ISDA),
        calendar: ql.Calendar = ql.UnitedStates(),
        allow_extrapolation: bool = False,
        valuation_date: datetime = None,
        interpolation_method: str = "Bilinear",
    ):
        # extract relevant parameters from options_data object if its passed
        if options_data is not None:
            vol_grid = options_data.volatility_grid
            valuation_date = options_data.valuation_date
        else:
            vol_grid = volatility_grid

        valuation_date_for_ql = _create_ql_date(valuation_date)

        # index must be an int64 index (possibly float, haven't tried!), so convert if needed
        if isinstance(vol_grid.columns, pd.core.indexes.base.Index):
            vol_grid.columns = vol_grid.columns.astype(int)

        # drop valdate if its in the index, QuantLib gives a garbage error about datesbeing sorted
        # unique, but it is also from valdate being in index
        idx = vol_grid.index == valuation_date
        vol_grid = vol_grid.loc[~idx, :]
        # convert expiration dates to quantlib Dates
        dates = pd.to_datetime((vol_grid.axes[0]).tolist())
        expiration_dates = list(map(_create_ql_date, dates))

        strikes = vol_grid.axes[1].tolist()
        vol_data = vol_grid.values.tolist()

        implied_vols = _create_ql_vol_grid(expiration_dates, strikes, vol_data)
        super().__init__(
            valuation_date_for_ql,
            calendar,
            expiration_dates,
            strikes,
            implied_vols,
            day_count,
        )
        if allow_extrapolation is True:
            self.enableExtrapolation()

        self.setInterpolation(interpolation_method)

    def generate_interpolated_volatility_dataframe(
        self,
        strike_step: int = 100,
        frequency_code: str = "7D",
        spot_price: float = None,
    ) -> pd.DataFrame:
        """Function to take an existing volatility surface, and interpolate across a specified
        interval of strikes (e.g. every 50 points) and days to maturity interval (e.g. every 7
        days) in order to come up with an entire surface, which can then be used for visualization
        or for storing data into a database.

        Args:
            vol_surface (ql.BlackVarianceSurface): Surface object that has already been
                fit to market data
            strike_step (int, optional): Resolution of the interpolation across the strike axis.
                Defaults to 100.
            frequency_code (str, optional): Resolution of the interpolation across the date axis.
                Defaults to "7D". Other options are
            available here: Other options available here:
                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
            spot_price (float, optional): Index value at time zero; if this is supplied, an
                additional column called moneyness will be generated. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe of interpolated volatility numbers
        """
        strike_and_tau_df = _extract_surface_data(
            self,
            strike_step,
            frequency_code,
        )
        calculated_vols = _calculate_vols(self, strike_and_tau_df)
        if spot_price is not None:
            calculated_vols["moneyness"] = calculated_vols["strikes"] / spot_price
        return calculated_vols
