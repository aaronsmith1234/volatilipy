=====
Usage
=====

To use Volatilipy in a project::

    import volatilipy as vol

To create a new OptionsData object, you first need to have a dataframe of options data. To be clear, you need to have a dataframe with 
columns for strike price, exercise date, option price, and option type (call or put, represented as c or p). Column names can be whatever 
you want, as you can map them to required columns later ::

    raw_option_data = pd.read_csv(
    "tests/test_support_files/dummyOptionsData.csv",
    parse_dates=["quote_date", "expiration"],
    )

Next, you can create the OptionsData object by passing the dataframe and a date object to the OptionsData constructor ::

    valdate = date(year=2020, month=6, day=30)
    options_data = vp.OptionsData(valdate, raw_option_data)