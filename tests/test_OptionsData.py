# This test file is intended to test the OptionsData object and its various methods

from datetime import datetime
from io import StringIO

import pandas as pd
import pytest

import volatilipy as vp

# Note that the code is intended to be able to read options data in a variety of formats, as long as
# a minimal set of columns exist. This is one format to test


@pytest.fixture
def dummy_position_data_format1():
    positionData = StringIO(
        """optionID,sTrIke,EXERCISEDATE,positionType,units,OPT_PX,option_type,mid_eod
S01-00161,3034.68,10/15/2020,long,1154.63167,205.15,C,203.15
"""
    )
    positionData = pd.read_csv(positionData)
    positionData["sTrIke"] = positionData["sTrIke"].astype(float)
    positionData["units"] = positionData["units"].astype(float)
    positionData["EXERCISEDATE"] = positionData["EXERCISEDATE"].astype("datetime64[ns]")
    return positionData


@pytest.fixture
def dummy_options_data_format1(dummy_position_data_format1) -> vp.OptionsData:
    options_data = vp.OptionsData(
        valuation_date=datetime(year=2020, month=6, day=30),
        option_positions=dummy_position_data_format1,
    )
    return options_data


# Note that the code is intended to be able to read options data in a variety of formats, as long as
# a minimal set of columns exist. This is a different format to test


@pytest.fixture
def dummy_options_data_format2() -> vp.OptionsData:
    raw_options_df = pd.read_csv(
        "tests/test_support_files/dummyOptionsData.csv",
        parse_dates=["quote_date", "expiration"],
    )
    dummy_options_data = vp.OptionsData(datetime(2030, 12, 31), raw_options_df)
    return dummy_options_data


@pytest.fixture
def dummy_rfr_data():
    rateData = StringIO(
        """date_index,SPOT_RATE_EFF_ANN
            10/13/2020,0.00061346
            10/14/2020,0.000612324
            10/15/2020,0.000611208
            10/16/2020,0.000610114
            10/17/2020,0.000609039
            """
    )
    rateData = pd.read_csv(rateData)
    rateData["date_index"] = rateData["date_index"].astype("datetime64[ns]")
    rateData.set_index("date_index", inplace=True)
    rateData["SPOT_RATE_EFF_ANN"] = rateData["SPOT_RATE_EFF_ANN"].astype(float)
    return rateData


def test_implied_vol_calc_call(
    dummy_rfr_data, dummy_options_data_format1: vp.OptionsData
):
    spx = 3100.29
    divs = 0.019

    columns_to_rename_for_implied_vol_calc = {
        "sTrIke": "strike",
        "EXERCISEDATE": "exercise_date",
        "units": "option_units",
        "OPT_PX": "option_price",
    }

    dummy_options_data_format1.solve_for_implied_vol(
        index_values=spx,
        dividend_yields=divs,
        risk_free_rates=dummy_rfr_data,
        columns_to_rename=columns_to_rename_for_implied_vol_calc,
        rate_column_name="SPOT_RATE_EFF_ANN",
    )

    assert dummy_options_data_format1.option_positions.loc[
        0, "implied_vol"
    ] == pytest.approx(0.272468)


def test_implied_vol_calc_put(
    dummy_rfr_data, dummy_options_data_format1: vp.OptionsData
):
    spx = 3100.29
    divs = 0.019
    dummy_options_data_format1.option_positions["option_type"] = "p"
    columns_to_rename_for_implied_vol_calc = {
        "sTrIke": "strike",
        "EXERCISEDATE": "exercise_date",
        "units": "option_units",
        "OPT_PX": "option_price",
    }

    dummy_options_data_format1.solve_for_implied_vol(
        index_values=spx,
        dividend_yields=divs,
        risk_free_rates=dummy_rfr_data,
        columns_to_rename=columns_to_rename_for_implied_vol_calc,
        rate_column_name="SPOT_RATE_EFF_ANN",
    )

    assert dummy_options_data_format1.option_positions.loc[
        0, "implied_vol"
    ] == pytest.approx(0.347316855)


def test_implied_vol_calc_nonsensical(
    dummy_rfr_data, dummy_options_data_format1: vp.OptionsData
):
    spx = 3100.29
    divs = 0.019
    dummy_options_data_format1.option_positions["OPT_PX"] = 500000000
    columns_to_rename_for_implied_vol_calc = {
        "sTrIke": "strike",
        "EXERCISEDATE": "exercise_date",
        "units": "option_units",
        "OPT_PX": "option_price",
    }

    dummy_options_data_format1.solve_for_implied_vol(
        index_values=spx,
        dividend_yields=divs,
        risk_free_rates=dummy_rfr_data,
        columns_to_rename=columns_to_rename_for_implied_vol_calc,
        rate_column_name="SPOT_RATE_EFF_ANN",
    )

    assert (
        dummy_options_data_format1.option_positions.loc[0, "implied_vol"]
        == "nonsensical"
    )


def test_implied_vol_calc_wrongOptType(
    dummy_rfr_data, dummy_options_data_format1: vp.OptionsData
):
    spx = 3100.29
    divs = 0.019
    dummy_options_data_format1.option_positions["option_type"] = "burrito"
    columns_to_rename_for_implied_vol_calc = {
        "sTrIke": "strike",
        "EXERCISEDATE": "exercise_date",
        "units": "option_units",
        "OPT_PX": "option_price",
    }
    with pytest.raises(ValueError):
        dummy_options_data_format1.solve_for_implied_vol(
            index_values=spx,
            dividend_yields=divs,
            risk_free_rates=dummy_rfr_data,
            columns_to_rename=columns_to_rename_for_implied_vol_calc,
            rate_column_name="SPOT_RATE_EFF_ANN",
        )


def test_implied_vol_calc_alternate_price_def(
    dummy_rfr_data, dummy_options_data_format1: vp.OptionsData
):
    spx = 3100.29
    divs = 0.019

    columns_to_rename_for_implied_vol_calc = {
        "sTrIke": "strike",
        "EXERCISEDATE": "exercise_date",
        "units": "option_units",
        "mid_eod": "option_price",
    }

    dummy_options_data_format1.solve_for_implied_vol(
        index_values=spx,
        dividend_yields=divs,
        risk_free_rates=dummy_rfr_data,
        columns_to_rename=columns_to_rename_for_implied_vol_calc,
        rate_column_name="SPOT_RATE_EFF_ANN",
    )
    assert dummy_options_data_format1.option_positions.loc[
        0, "implied_vol"
    ] == pytest.approx(0.2694103)


def test_vol_grid(dummy_options_data_format2):
    dummy_options_data_format2.calculate_volatility_grid()
    # pick some random points to confirm it works
    assert dummy_options_data_format2.volatility_grid[2600][0] == pytest.approx(
        1.610949288
    )
    assert dummy_options_data_format2.volatility_grid[3050][28] == pytest.approx(
        1.749559383
    )


def test_vol_grid_puts(dummy_options_data_format2):
    # Note that this file, while having the same format as a CBOE options with calcs file, has garbage data.
    dummy_options_data_format2.calculate_volatility_grid(fit_on_calls_or_puts="puts")
    # pick some random points to confirm it works
    assert dummy_options_data_format2.volatility_grid[2200][5] == pytest.approx(
        0.239444034
    )
    assert dummy_options_data_format2.volatility_grid[3800][12] == pytest.approx(
        1.804490279
    )


def test_vol_grid_both(dummy_options_data_format2):
    # Note that this file, while having the same format as a CBOE options with calcs file, has garbage data.
    dummy_options_data_format2.calculate_volatility_grid(fit_on_calls_or_puts="both")
    # pick some random points to confirm it works
    assert dummy_options_data_format2.volatility_grid[4900][11] == pytest.approx(
        1.259525493
    )
    assert dummy_options_data_format2.volatility_grid[1400][6] == pytest.approx(
        2.161576293
    )


def test_file_export(dummy_options_data_format2: vp.OptionsData):
    dummy_options_data_format2.calculate_volatility_grid(
        fit_on_calls_or_puts="both",
        to_file_filename="tests/test_support_files/testOptionsDataExport.csv",
    )
