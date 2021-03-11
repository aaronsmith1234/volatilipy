"""Helper functions for the package"""
from collections import namedtuple
from datetime import date, datetime
from typing import Tuple, Union

import pandas as pd
import QuantLib as ql
from numpy import float64


# helper functions for the volatility surface class
def _extract_surface_data(
    vol_surface: ql.BlackVarianceSurface,
    strike_step: int = 100,
    frequency_code: str = "7D",
) -> pd.DataFrame:
    """Function to take a volatility surface and render a dataframe of strikes and taus that
    correspond to that dataframe. E.g. if the surface goes from strikes 500 to 3500 in steps of
    50, you can use this dataframe to create a dataframe that starts at 500, ends at 3500, but at
    steps of 5 or 10 or whatever (e.g. strike_step variable). Same can be done for the maturity
    dates/expiry dates. This will then be filled in by interpolating across the surface object.

    Args:
        vol_surface (ql.BlackVarianceSurface): Surface from which you want to extract surface data;
            e.g. moneyness max/min and tau max/min
        strike_step (int, optional): What steps you want the new strike dataframe to be in.
            Defaults to 100.
        frequency_code (str, optional): What steps you want the new tau/expiry date dataframe to
            be in. Defaults to "7D".
        Other options available here:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    Returns:
        pd.DataFrame: [description]
    """
    strike_df = _generate_strike_df(vol_surface, strike_step)
    date_df = _generate_date_df(vol_surface, frequency_code)
    cartesian_product = pd.MultiIndex.from_product(
        [strike_df.values.tolist(), date_df["tau"].values.tolist()],
        names=["strikes", "tau"],
    )
    merged_df = pd.DataFrame(index=cartesian_product).reset_index()
    merged_df = merged_df.merge(
        date_df[["expiryDate", "tau", "days_to_maturity"]], on="tau"
    )
    merged_df = merged_df[merged_df["tau"] != 0]
    return merged_df


def _generate_strike_df(
    vol_surface: ql.BlackVarianceSurface, strike_step: int
) -> pd.DataFrame:
    """Function to take the max/min strikes from a surface object and create new
    datapoints at a given step/increment

    Args:
        vol_surface (ql.BlackVarianceSurface): Surface to extract data from
        strike_step (int): How far apart you want the new strikes to be

    Returns:
        pd.DataFrame: Dataframe consisting of new strikes at the given interval
    """
    strike_list = []
    temp_strike = vol_surface.minStrike()
    strike_list.append(temp_strike)

    while temp_strike < vol_surface.maxStrike() - strike_step:
        temp_strike += strike_step
        strike_list.append(temp_strike)

    strike_list.append(vol_surface.maxStrike())
    strike_df = pd.Series(strike_list)
    return strike_df


def _generate_date_df(
    vol_surface: ql.BlackVarianceSurface,
    frequency_code: str = "7D",
) -> pd.DataFrame:
    """Function to take a surface with an implied max/min date and generate a list of taus
    at the specified intervals.

    Args:
        vol_surface (ql.BlackVarianceSurface): Surface to extract start/end dates from
        frequency_code (str, optional): What steps you want the new dataframe to have.
            Defaults to "7D".
        Other options available here:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    Returns:
        pd.DataFrame: Dataframe with new taus!
    """
    date_df = pd.date_range(
        start=vol_surface.referenceDate().to_date(),
        end=vol_surface.maxDate().to_date(),
        freq=frequency_code,
    )
    date_df = date_df.to_frame(index=True, name="expiryDate")
    date_df["QuantlibDate"] = date_df.apply(_datetime_to_quantlib_date, axis=1)
    date_df["tau"] = date_df.apply(
        _calculate_tau, args=(vol_surface.referenceDate(), ql.ActualActual()), axis=1
    )
    date_df["days_to_maturity"] = date_df.apply(
        _calculate_dtm, args=(vol_surface.referenceDate(), ql.ActualActual()), axis=1
    )
    return date_df


def _calculate_tau(
    end_date: ql.Date,
    start_date: ql.Date,
    day_count_convention: ql.DayCounter,
) -> float:
    """Function to calculate a tau/difference in time from Quantlib objects

    Args:
        end_date (ql.Date): End date
        start_date (ql.Date): Start date
        day_count_convention (ql.DayCounter): What basis of days to use;
            ActualActual, 30/360, etc.

    Returns:
        float: tau in years of the difference between the two dates
    """

    if isinstance(end_date, pd.core.series.Series):
        end_date = end_date["QuantlibDate"]
    return day_count_convention.yearFraction(start_date, end_date)


def _calculate_dtm(
    end_date: ql.Date,
    start_date: ql.Date,
    day_count_convention: ql.DayCounter,
) -> float:
    """Function to calculate a difference in days between two dates

    Args:
        end_date (ql.Date): End date
        start_date (ql.Date): Start date
        day_count_convention (ql.DayCounter): What basis of days to use;
            ActualActual, 30/360, etc.

    Returns:
        float: tau in years of the difference between the two dates
    """

    if isinstance(end_date, pd.core.series.Series):
        end_date = end_date["QuantlibDate"]
    return day_count_convention.dayCount(start_date, end_date)


def _create_ql_date(input_date: Union[date, datetime]) -> ql.Date:
    """Method to turn a datetime into a Quantlib Date

    Args:
        input_date (Union[date, datetime]): Date or datetime object

    Returns:
        ql.Date: Same date represented as a Quantlib Date
    """
    return ql.Date(input_date.day, input_date.month, input_date.year)


def _calculate_vols(
    vol_surface: ql.BlackVarianceSurface, strike_and_tau_df: pd.DataFrame
) -> pd.DataFrame:
    temp_df = strike_and_tau_df
    temp_df["volatility"] = strike_and_tau_df.apply(
        lambda row: vol_surface.blackVol(row.tau, row.strikes), axis=1
    )
    return temp_df


def _create_ql_vol_grid(
    expiration_dates: list, strikes: list, vol_data: list
) -> ql.Matrix:
    """Function to take lists of expirys, strikes, and volatility data, and fit it into a
    quantlib matrix which can then be used to create a BlackVarianceSurface

    Args:
        expiration_dates (list): [description]
        strikes (list): [description]
        vol_data (list): [description]

    Returns:
        ql.Matrix: [description]
    """
    implied_vols = ql.Matrix(len(strikes), len(expiration_dates))
    for i in range(implied_vols.rows()):
        for j in range(implied_vols.columns()):
            # need to transpose since thats the format that quantlib expects
            # todo add a parameter that can make this transpose or not,
            # depending on what source data shape is
            implied_vols[i][j] = vol_data[j][i]
    return implied_vols


# Helper functions for the implied vol object
def _get_option_rfr(
    risk_free_rates: pd.DataFrame,
    position: namedtuple,
    day_count_convention: ql.DayCounter,
    rate_column_name: str,
) -> float:
    """Method to pull market data that is specific to one option (namely volatility
    and a risk free rate)

    Args:
        risk_free_rates (pd.DataFrame): Dataframe containing risk free rates. Used to
            pull the risk free rate corresponding to the date of maturity of the option
        position (namedtuple): A named tuple that represents one option position; used to
            retrieve the maturity date and strike
        day_count_convention (QuantLibday_count_convention): Method to determine how to count days
        between two dates
    Returns:
        risk_free_rate: The risk free rate
    """

    rate = ql.SimpleQuote(risk_free_rates[rate_column_name][position.exercise_date])
    risk_free_rate = ql.FlatForward(
        0, ql.TARGET(), ql.QuoteHandle(rate), day_count_convention
    )
    return (risk_free_rate,)


def _setup_quantlib_economy(
    valuation_date: datetime, index_values: float64, dividend_yields: float64
) -> Tuple:
    """Method to setup the appropriate economy variables for quantlib, takes original datatypes
        and returns them in quantlib compatible objects.
    Args:
        valuation_date (datetime): [description]
        index_values (np.float64): [description]
        dividend_yields (np.float64): [description]
    Returns:
        [type]: [description]
    """

    valuation_date_for_quantlib = ql.Date(
        valuation_date.day, valuation_date.month, valuation_date.year
    )
    ql.Settings.instance().evaluationDate = valuation_date_for_quantlib
    day_count_convention = ql.ActualActual()

    spot = index_values
    spot_quote = ql.QuoteHandle(ql.SimpleQuote(spot))

    div_yield_rate = dividend_yields
    div_yield_rate_quote = ql.QuoteHandle(ql.SimpleQuote(div_yield_rate))
    dividend_yield = ql.YieldTermStructureHandle(
        ql.FlatForward(0, ql.TARGET(), div_yield_rate_quote, day_count_convention)
    )
    return (
        day_count_convention,
        spot,
        spot_quote,
        dividend_yield,
        div_yield_rate,
        valuation_date_for_quantlib,
    )


def _filter_option_types(
    option_df: pd.DataFrame, calls_or_puts: str, column_name_mapping: dict
) -> pd.DataFrame:
    """Method to filter out a dataframe to fit the surface based on only calls
    or only puts; any other entry will include both calls and puts

    Args:
        option_df (pd.DataFrame): Input dataframe of options
        calls_or_puts (str): Fit surface based on calls or puts.

    Returns:
        pd.DataFrame: dataframe with only the specified option types remaining
    """
    if calls_or_puts.lower() == "calls":
        filtered_df = option_df[option_df[column_name_mapping["option_type"]] == "C"]
    elif calls_or_puts.lower() == "puts":
        filtered_df = option_df[option_df[column_name_mapping["option_type"]] == "P"]
    else:
        filtered_df = option_df
    return filtered_df


def _drop_zero_vols(
    filtered_options: pd.DataFrame, column_name_mapping: dict
) -> pd.DataFrame:
    """Function to remove all options with zero volatilies from the option list

    Args:
        filtered_options (pd.DataFrame): Dataframe of options

    Returns:
        pd.DataFrame: Dataframe without any rows where the implied volatility is zero
    """
    no_more_zero_vols = filtered_options.drop(
        filtered_options[
            filtered_options[column_name_mapping["implied_volatility"]] == 0
        ].index
    )

    no_more_zero_vols = no_more_zero_vols.drop(
        no_more_zero_vols[
            no_more_zero_vols[column_name_mapping["implied_volatility"]]
            == "nonsensical"
        ].index
    )
    # only have values in this now
    no_more_zero_vols[column_name_mapping["implied_volatility"]] = no_more_zero_vols[
        column_name_mapping["implied_volatility"]
    ].astype("float64")

    return no_more_zero_vols


def _average_across_options(
    option_df: pd.DataFrame, column_name_mapping: dict
) -> pd.DataFrame:
    """Function to take a list of options and average across the strike and
    expiration dates to get average implied volatilies. This is needed since
    some option request files (e.g. SPX) include both SPX and SPXW options,
    where the strike and expiration can be the same, but they can have different
    prices due to different settlement times. An alternative would be to only
    keep the one with more market liquidity, or do some type of weighted
    average, but for now this provides a decent fit.

    Args:
        option_df (pd.DataFrame): Dataframe of options data

    Returns:
        pd.DataFrame: A dataframe with row indices of expiration dates, and
        column headers of strike prices, with the dataframe entries being
        average implied volatilities
    """
    averaged_vols = (
        option_df.groupby(
            [column_name_mapping["strike"], column_name_mapping["expiration"]]
        )
        .mean()
        .reset_index()
    )

    averaged_vols_pivoted = averaged_vols.pivot_table(
        index=column_name_mapping["expiration"],
        columns=column_name_mapping["strike"],
        values=column_name_mapping["implied_volatility"],
    )
    _sort_df(averaged_vols_pivoted)
    return averaged_vols_pivoted


def _drop_sparse_columns(option_df: pd.DataFrame, threshold=0.75) -> pd.DataFrame:
    """Method to drop any columns that are x% NaN (i.e. null/blank) or less. Needed in
    this case since QuantLib requires surfaces to be entirely filled in or else it
    can't create Black Volatility surface objects.

    Args:
        option_df (pd.DataFrame): Dataframe of option implied volatilies
        threshold (float, optional): What percent of the column must be filled
        in in order to keep as part of the dataframe. Defaults to 0.75.

    Returns:
        pd.DataFrame: Dataframe with only columns filled in up to the specified threshold
    """
    not_as_sparse_columns = option_df.dropna(
        axis=1, thresh=round(len(option_df.index) * threshold), how="any"
    )
    return not_as_sparse_columns


def _fill_in_grid(option_df: pd.DataFrame) -> pd.DataFrame:
    """Function to interpolate data for missing datapoints. I'm doing simple linear
    interpolation; in reality, a 2d interpolation would probably give a better fit,
    but the current fit provided by this method is fine, assuming your columns are
    mostly filled in. The interpolation is first done across columns (i.e. differing
    strikes on same expiration date), and then across rows
    (i.e. differing expirations for the same strike)

    Args:
        option_df (pd.DataFrame): Dataframe of implied volatilies

    Returns:
        pd.DataFrame: Dataframe that is mostly filled in; we aren't doing
        extrapolation, so any edges that are missing values are still going to be blank!
    """
    interpolate_across_columns = option_df.interpolate(axis=1)
    interpolate_across_rows = interpolate_across_columns.interpolate(axis=0)
    return interpolate_across_rows


def _drop_valdate_row(option_df: pd.DataFrame, valuation_date: date) -> pd.DataFrame:
    """Function to drop any rows with the valuation date as the expiration date;
    this is due to Quantlib not allowing a surface to be fit with a row for
    the valuation date (e.g. time to expiry 0)

    Args:
        option_df (pd.DataFrame): Dataframe of implied volatilities
        valuation_date (date): Date to drop!

    Returns:
        pd.DataFrame: Original dateframe but initial row missing
    """
    if (
        pd.Timestamp(valuation_date) in option_df.index
        or valuation_date in option_df.index
    ):
        cleaned_df = option_df.drop([valuation_date])
    else:
        cleaned_df = option_df
    return cleaned_df


def _sort_df(option_df: pd.DataFrame):
    option_df = option_df.sort_index(axis=0)
    option_df = option_df.sort_index(axis=1)


def _datetime_to_quantlib_date(
    input_datetime: Union[datetime, pd.Series],
    datetime_column_name: str = "exercise_date",
) -> ql.Date:
    """Function to take a datetime object, return a quantlib object representing the same date

    Args:
        input_datetime (Union[datetime, pd.Series]): Either a datetime object or a pandas
            series object that contains datetime objects

    Returns:
        ql.Date: The same dates, represented as Quantlib objects
    """
    if isinstance(input_datetime, pd.core.series.Series):
        cols = input_datetime.shape[0]
        if cols > 1:
            return_date = ql.Date(
                input_datetime.loc[datetime_column_name].day,
                input_datetime.loc[datetime_column_name].month,
                input_datetime.loc[datetime_column_name].year,
            )
        else:
            return_date = ql.Date(
                input_datetime[0].day,
                input_datetime[0].month,
                input_datetime[0].year,
            )
    else:
        return_date = ql.Date(
            input_datetime.day,
            input_datetime.month,
            input_datetime.year,
        )
    return return_date


def _create_quantlib_option(dataframe_row: pd.Series) -> ql.EuropeanOption:
    if dataframe_row.option_type.lower() == "c":
        call_or_put = ql.Option.Call
    elif dataframe_row.option_type.lower() == "p":
        call_or_put = ql.Option.Put
    else:
        call_or_put = "error!"
        raise ValueError("Option type is not C or P")

    return ql.EuropeanOption(
        ql.PlainVanillaPayoff(call_or_put, dataframe_row.strike),
        ql.EuropeanExercise(dataframe_row.QuantlibDate),
    )


def _calculate_tau(
    dataframe_row: pd.Series,
    valuation_date_for_quantlib: ql.Date,
    day_count_convention: ql.DayCounter,
) -> float:
    tau = day_count_convention.yearFraction(
        valuation_date_for_quantlib, dataframe_row.QuantlibDate
    )
    return tau


def _get_option_rfr_for_implied_vol(
    dataframe_row: pd.Series,
    risk_free_rates: pd.DataFrame,
    day_count_convention: ql.DayCounter,
    rate_column_name: str,
):
    rate_quote = ql.SimpleQuote(
        risk_free_rates[rate_column_name][dataframe_row.exercise_date]
    )
    risk_free_rate = ql.FlatForward(
        0, ql.TARGET(), ql.QuoteHandle(rate_quote), day_count_convention
    )
    return risk_free_rate


def _create_bsm_process_with_dummy_vol(
    dataframe_row: pd.Series,
    spot_quote: ql.QuoteHandle,
    dividend_yield: ql.YieldTermStructureHandle,
    day_count_convention: ql.DayCounter,
) -> ql.AnalyticEuropeanEngine:
    vol_quote = ql.BlackConstantVol(
        0,
        ql.TARGET(),
        ql.QuoteHandle(ql.SimpleQuote(0.2222)),
        day_count_convention,
    )
    # .2222 is a dummy vol just to let quantlib not yell at us!

    process = ql.BlackScholesMertonProcess(
        spot_quote,
        dividend_yield,
        ql.YieldTermStructureHandle(dataframe_row.risk_free_rate),
        ql.BlackVolTermStructureHandle(vol_quote),
    )

    return process


def _create_bsm_engine(dataframe_row: pd.Series) -> ql.AnalyticEuropeanEngine:
    return ql.AnalyticEuropeanEngine(dataframe_row.bsm_process)


def _set_pricing_engine(dataframe_row: pd.Series) -> None:
    dataframe_row.position_object.setPricingEngine(dataframe_row.bsm_engine)


def _output_rfr(dataframe_row: pd.Series):
    return dataframe_row.risk_free_rate.zeroRate(1, ql.Continuous, ql.Annual).rate()


def _calculate_implied_vol(dataframe_row: pd.Series) -> float:
    try:
        implied_vol = dataframe_row.position_object.impliedVolatility(
            dataframe_row.option_price, dataframe_row.bsm_process
        )
    except RuntimeError:
        implied_vol = "nonsensical"
    return implied_vol
