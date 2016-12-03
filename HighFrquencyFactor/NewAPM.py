def APM(stocks,end):
    window = 33
    AllRes = pd.DataFrame()
    start = "{:%Y-%m-%d}".format(datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=window))
    prices = get_price(stocks, start_date=start, end_date=end, frequency='1d')['OpeningPx']
    APM = APMRaw(stocks,end)
    twdreturn = (np.log(prices.iloc[-1]/prices.iloc[0])).T
    DataAll = pd.concat([twdreturn,APM],axis = 1)
    #DataAll.columns[-1] = 'Index'
    #DataAll = 
    DataAll = DataAll.dropna()
    regr = linear_model.LinearRegression()
    try:
        regr.fit(np.matrix(DataAll.ix[:,0]).T, np.matrix(DataAll.ix[:,1]).T)
        residuals = regr.predict(np.matrix(DataAll.ix[:,0]).T) - np.matrix(DataAll.ix[:,1]).T
        residuals = pd.DataFrame(data = residuals, index = DataAll.index.values)
        residuals.columns = ['NewAPM']
    except ValueError:
        print("No data for this stock")
    return residuals.ix[:,0]
