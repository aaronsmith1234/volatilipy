"""Class designed to hold a dataframe of options, and then be able to do various activities;
e.g. calculate implied volatilities, create a grid of volatilies (combining across options)
"""
from datetime import datetime

import numpy as np
import pandas as pd

from .helper_functions import (
    _setup_quantlib_economy,
    _filter_option_types,
    _drop_zero_vols,
    _average_across_options,
    _drop_sparse_columns,
    _fill_in_grid,
    _drop_valdate_row,
    _sort_df,
    _create_quantlib_option,
    _calculate_tau,
    _datetime_to_quantlib_date,
    _get_option_rfr_for_implied_vol,
    _set_pricing_engine,
    _create_bsm_process_with_dummy_vol,
    _output_rfr,
    _calculate_implied_vol,
    _create_bsm_engine,
)


class OptionsData:
    """Object to take a dataframe of options (called options_option_positions) and backsolve for
    what the implied volatility is under Black Scholes Merton

    Args:
        valuation_date (datetime): Date for which you want the option valued
        option_positions (pd.DataFrame): Dataframe containing options
    """

    def __init__(
        self,
        valuation_date: datetime,
        option_positions: pd.DataFrame,
    ) -> None:
        self.valuation_date = valuation_date
        self.option_positions = option_positions
        self.volatility_grid = None

    def solve_for_implied_vol(
        self,
        index_values: np.float64,
        dividend_yields: np.float64,
        risk_free_rates: pd.DataFrame,
        columns_to_rename: dict = {
            "strike": "strike",
            "exercise_date": "exercise_date",
            "mid_eod": "option_price",
            "option_type": "option_type",
        },
        rate_column_name: str = "spot_rate_eff_ann",
    ) -> None:
        """This is where the magic happens! Backsolving for implied volatility
        Note that the following columns must exist in the dataframe in order for the pricing
        to work. If needed, use the columns_to_rename in order to rename the columns temporarily
        for the volatility calculation (i.e. at the end of the pricing routine, the columns
        will be renamed to their original values):


        Args:
            index_values (np.float64): Value of underlying index as of the valuation date
            dividend_yields (np.float64): Dividend yield for the underlying index as of the
                valuation date
            risk_free_rates (pd.DataFrame): Dataframe containing risk free rates to use in BSM
            volatilities (pd.DataFrame): Dateframe with some beautiful option volatilities.
            columns_to_rename (dict): Dictionary with columns to rename from source file.
                Refer to function declaration in the values fields for columns that need to exist.
                At the end of this method, the dictionary is inversed, and all column names will be
                returned to their original values
        Returns:
            pd.DataFrame: Dataframe containing the original columns, plus the parameters used in
                BSM for each option
        """

        (
            day_count_convention,
            spot,
            spot_quote,
            dividend_yield,
            div_yield_rate,
            valuation_date_for_quantlib,
        ) = _setup_quantlib_economy(self.valuation_date, index_values, dividend_yields)

        self.option_positions.rename(columns=columns_to_rename, inplace=True)
        self.option_positions["spot"] = spot
        self.option_positions["dividendYield"] = div_yield_rate
        self.option_positions["QuantlibDate"] = self.option_positions.apply(
            _datetime_to_quantlib_date, axis=1
        )
        self.option_positions["position_object"] = self.option_positions.apply(
            _create_quantlib_option, axis=1
        )
        self.option_positions["tau"] = self.option_positions.apply(
            _calculate_tau,
            args=(valuation_date_for_quantlib, day_count_convention),
            axis=1,
        )

        self.option_positions["risk_free_rate"] = self.option_positions.apply(
            _get_option_rfr_for_implied_vol,
            args=(risk_free_rates, day_count_convention, rate_column_name),
            axis=1,
        )

        self.option_positions["bsm_process"] = self.option_positions.apply(
            _create_bsm_process_with_dummy_vol,
            args=(spot_quote, dividend_yield, day_count_convention),
            axis=1,
        )

        self.option_positions["bsm_engine"] = self.option_positions.apply(
            _create_bsm_engine,
            axis=1,
        )

        self.option_positions.apply(_set_pricing_engine, axis=1)

        self.option_positions["riskFreeRate"] = self.option_positions.apply(
            _output_rfr, axis=1
        )

        self.option_positions["implied_vol"] = self.option_positions.apply(
            _calculate_implied_vol, axis=1
        )

        columns_to_rename = {v: k for k, v in columns_to_rename.items()}
        self.option_positions.rename(columns=columns_to_rename, inplace=True)

        self.option_positions.drop(
            [
                "position_object",
                "risk_free_rate",
                "bsm_process",
                "bsm_engine",
                "QuantlibDate",
            ],
            axis=1,
            inplace=True,
        )

    def calculate_volatility_grid(
        self,
        fit_on_calls_or_puts: str = "calls",
        to_file_filename: str = None,
        column_name_mapping: dict = {
            "option_type": "option_type",
            "implied_volatility": "implied_volatility_1545",
            "strike": "strike",
            "expiration": "expiration",
        },
    ) -> None:
        """Method to process a dataframe of options and return a grid of implied volatilies by
        expiration date and strike, which can then be fed into QuantLib to fit a volatility surface.
        An example of this data that you can buy this data from
        https://datashop.cboe.com/option-quotes-end-of-day-with-calcs-subscription
        or https://datashop.cboe.com/option-quotes-end-of-day-with-calcs

        Args:
            calls_or_puts (str, optional): Whether to fit the implied volatilties
            from calls or puts, or both. Defaults to "calls".
            to_file_filename(str, optional): filepath where to save the finalized
            file, useful for looking at results. Should be a csv file. Defaults to None
        Returns:
            pd.DataFrame: dataframe of implied volatilities
        """
        column_name_mapping_defaults = {
            "option_type": "option_type",
            "implied_volatility": "implied_volatility_1545",
            "strike": "strike",
            "expiration": "expiration",
        }

        column_name_mapping_temp = {}
        column_name_mapping_temp.update(column_name_mapping_defaults)
        column_name_mapping_temp.update(column_name_mapping)
        column_name_mapping = column_name_mapping_temp

        filtered_on_option_type = _filter_option_types(
            self.option_positions, fit_on_calls_or_puts, column_name_mapping
        )
        all_nonzero_vols = _drop_zero_vols(filtered_on_option_type, column_name_mapping)
        averaged_vols = _average_across_options(all_nonzero_vols, column_name_mapping)
        less_sparse_grid = _drop_sparse_columns(averaged_vols)
        mostly_filled_in_grid = _fill_in_grid(less_sparse_grid)
        complete_grid = _drop_sparse_columns(mostly_filled_in_grid, 1)
        vol_grid_for_quantlib = _drop_valdate_row(complete_grid, self.valuation_date)
        _sort_df(vol_grid_for_quantlib)
        if to_file_filename is not None:
            vol_grid_for_quantlib.to_csv(to_file_filename)

        self.volatility_grid = vol_grid_for_quantlib
