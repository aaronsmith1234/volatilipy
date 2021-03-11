# This test file is intended to test the Volatility Surface object and its various methods

from datetime import datetime

import pandas as pd
import pytest

import volatilipy as vp


@pytest.fixture
def valuation_date():
    return datetime(2030, 12, 31)


@pytest.fixture
def options_data(valuation_date):
    raw_options_df = pd.read_csv(
        "tests/test_support_files/dummyOptionsData.csv",
        parse_dates=["quote_date", "expiration"],
    )
    options_data = vp.OptionsData(valuation_date, raw_options_df)
    options_data.calculate_volatility_grid()
    return options_data


@pytest.fixture
def mock_vol_surface(options_data: vp.OptionsData) -> vp.VolatilitySurface:

    vol_surface = vp.VolatilitySurface(options_data, interpolation_method="Bicubic")
    return vol_surface


def test_render_vol_surface(options_data: vp.OptionsData):

    vol_surface = vp.VolatilitySurface(options_data)
    assert vol_surface.blackVol(1, 3123) == pytest.approx(1.0742223)


def test_vol_surface_extrapolation_fails(options_data: vp.OptionsData):

    vol_surface = vp.VolatilitySurface(options_data)
    with pytest.raises(RuntimeError):
        vol_surface.blackVol(10, 3000)


def test_vol_surface_time_extrapolation_works(options_data: vp.OptionsData):

    vol_surface = vp.VolatilitySurface(options_data, allow_extrapolation=True)
    assert vol_surface.blackVol(10, 3000) == pytest.approx(1.1993656)


def test_vol_surface_interpolation_method(options_data: vp.OptionsData):

    vol_surface = vp.VolatilitySurface(options_data, interpolation_method="Bicubic")
    assert vol_surface.blackVol(1, 3123) == pytest.approx(0.997225678)


def test_interp_df(mock_vol_surface: vp.VolatilitySurface):
    interpolated = mock_vol_surface.generate_interpolated_volatility_dataframe()
    assert interpolated["strikes"].min() == 2300
    assert interpolated[interpolated["strikes"] == 2350].size == 0
    assert interpolated["strikes"].max() == 5600
    assert interpolated["tau"].min() == pytest.approx(0.01917808219178082)
    assert interpolated["tau"].max() == pytest.approx(2.950684)


def test_interp_df_strike_interp(mock_vol_surface):
    interpolated = mock_vol_surface.generate_interpolated_volatility_dataframe(50)
    assert interpolated[interpolated["strikes"] == 2350]["strikes"].size == 154


def test_interp_df_dtm_interp(mock_vol_surface):
    interpolated = mock_vol_surface.generate_interpolated_volatility_dataframe(
        frequency_code="21D"
    )
    assert interpolated["days_to_maturity"].min() == 21
    assert interpolated[interpolated["days_to_maturity"] == 7].size == 0


def test_interp_df_no_spot(mock_vol_surface):
    interpolated = mock_vol_surface.generate_interpolated_volatility_dataframe()
    with pytest.raises(KeyError):
        interpolated["moneyness"]


def test_interp_df_with_spot(mock_vol_surface):
    interpolated = mock_vol_surface.generate_interpolated_volatility_dataframe(
        spot_price=2500
    )
    assert interpolated["moneyness"].min() == pytest.approx(0.92)


def test_interp_df_with_different_spot(mock_vol_surface):
    interpolated = mock_vol_surface.generate_interpolated_volatility_dataframe(
        spot_price=3500
    )
    assert interpolated["moneyness"].min() == pytest.approx(0.6571428571428571)
