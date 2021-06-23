=====
Usage
=====

OptionsData
--------------

To use Volatilipy in a project::

    import volatilipy as vp

To create a new ``OptionsData`` object, you first need to have a Pandas dataframe of options data. To be clear, you need to have a Pandas dataframe with 
columns for strike price, exercise date, option price, and option type (call or put, represented as c or p). Column names can be whatever 
you want, as you can map them to required columns later. There are a variety of market data providers where you can get this type of information.
Options exchanges typically have very fair rates for providing once a day feeds of option prices, and some even have implied volatilies already calculated for you. ::

    raw_option_data = pd.read_csv(
    "tests/test_support_files/dummyOptionsData.csv",
    parse_dates=["quote_date", "expiration"],
    )

Next, you can create the ``OptionsData`` object by passing the dataframe and a date object to the ``OptionsData`` constructor ::

    valdate = date(year=2020, month=6, day=30)
    options_data = vp.OptionsData(valdate, raw_option_data)

If you already have an implied volatilty column, then you can skip this step. But if not, and you need to solve for implied volatility,
you can use the inbuilt methods. This will generate a new column called "implied_vol" in your dataframe. See the tests for examples ::

    columns_to_rename_for_implied_vol_calc = {
        "sTrIke": "strike",
        "EXERCISEDATE": "exercise_date",
        "units": "option_units",
        "OPT_PX": "option_price",
    }

    options_data.solve_for_implied_vol(
        index_values=spx,
        dividend_yields=divs,
        risk_free_rates=rfr_data,
        columns_to_rename=columns_to_rename_for_implied_vol_calc,
        rate_column_name="SPOT_RATE_EFF_ANN",
    )

The final step to using an ``OptionsData`` object is to get the volatility grid. This is a dataframe of implied volatilities organized by expiration date on the rows,
and strike prices on the columns. The method ``OptionsData.calculate_volatility_grid`` will generate this grid for you; it also handles sparse data by dropping columns
that have few entries, and in cases where the data is mostly filled it, it will interpolate volatilies from nearby market data points. There are a few parameters
you can use to decide how you want the algorithm to behave; see the docstring.::

    options_data.calculate_volatility_grid()

At this point, you are ready to use your ``OptionsData`` object to calculate a volatility surface; buckle up. 

VolatilitySurface
--------------------------

Fill me in!