# Simple example of how to take create an options data object, fit the volatility surface,
# and then interpolate across given strike and expirations dates

# note that you'll need to move this to base directory so it can resolve the import correctly

from datetime import date

import pandas as pd

import volatilipy as vp

valdate = date(year=2020, month=6, day=30)
raw_vols = pd.read_csv(
    "tests/test_support_files/dummyOptionsData.csv",
    parse_dates=["quote_date", "expiration"],
)

options_data = vp.OptionsData(valdate, raw_vols)
options_data.calculate_volatility_grid(fit_on_calls_or_puts="both")

# optionally, output the grid to see what the vol surface looks like
options_data.volatility_grid.to_csv("grid.csv")

vol_surface = vp.VolatilitySurface(options_data, interpolation_method="Bicubic")
vol_data = vol_surface.generate_interpolated_volatility_dataframe(
    strike_step=50, frequency_code="7D", spot_price=3756.07
)

# optionally, output the vol data so you can see what the vol points are, and can visualize easily
vol_data.to_csv("surface.csv")
