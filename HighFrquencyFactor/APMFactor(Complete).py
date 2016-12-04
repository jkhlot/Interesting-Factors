def APMRaw(stocks,end):
    window = 20
    start = "{:%Y-%m-%d}".format(datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=window))
    indp = get_price('000001.XSHG', start_date=start, end_date=end, frequency='1m')['OpeningPx']        
    indreturns = np.log(indp/indp.shift(1))
    AllRes = pd.DataFrame(indreturns)
    for stock in list(stocks):
        #print(stock)
        price = get_price(stock, start_date=start, end_date=end, frequency='1m')['OpeningPx']
        preturns = np.log(price/price.shift(1))
        DataAll = pd.concat([preturns,indreturns],axis = 1)
        #DataAll.columns[-1] = 'Index'
        #DataAll = 
        DataAll = DataAll.dropna()
        regr = linear_model.LinearRegression()
        try:
            regr.fit(np.matrix(DataAll.ix[:,0]).T, np.matrix(DataAll.ix[:,1]).T)
            residuals = regr.predict(np.matrix(DataAll.ix[:,0]).T) - np.matrix(DataAll.ix[:,1]).T
            residuals = pd.DataFrame(data = residuals, index = DataAll.index.values)
            residuals.columns = [stock]
            AllRes = pd.concat([AllRes,residuals],axis = 1)
        except ValueError:
            print("No data for this stock")
    AllRes = AllRes.ix[:,1:-1]
    Mor_Res = pd.DataFrame()
    Aft_Res = pd.DataFrame()
    Mor_Res = AllRes.ix[pd.to_datetime(AllRes.index.values).hour <= 12,:]
    Aft_Res = AllRes.ix[pd.to_datetime(AllRes.index.values).hour > 12,:]
    Diff = pd.DataFrame(np.array(Mor_Res) - np.array(Aft_Res))
    Diff.columns = Mor_Res.columns
    return np.mean(Diff)/(np.std(Diff)*np.sqrt(window))

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
    
