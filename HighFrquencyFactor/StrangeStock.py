def Strange(stocks,date):  #We still need to furthur adjust this factor by buying stocks only when CSI is lower
    window = 50
    end = date
    start = "{:%Y-%m-%d}".format(datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=window))
    indp = get_price('000001.XSHG', start_date=start, end_date=end, frequency='1m')['OpeningPx']        
    if indp[-1] < indp[0]: #Only open when the index is heading lower
        indreturns = np.log(indp/indp.shift(1))
        indret = pd.DataFrame(indreturns)
        AllRes = pd.DataFrame()
        for stock in list(stocks):
            #print(stock)
            tempRes = []
            price = get_price(stock, start_date=start, end_date=end, frequency='1m')['OpeningPx']
            preturns = np.log(price/price.shift(1))
            DataAll = pd.concat([preturns,indret],axis = 1) #First column is stock return, second column is stock return
            DataAll = DataAll.dropna()
            DataAll['month'] = pd.to_datetime(DataAll.index.values).month
            DataAll['day'] = pd.to_datetime(DataAll.index.values).day
            umonths = list(set(pd.to_datetime(DataAll.index.values).month))
            udays = list(set(pd.to_datetime(DataAll.index.values).day))
            for month in umonths:
                for day in udays:
                    temp = DataAll[(DataAll.month.values==month)&(DataAll.day.values==day)]
                    if ~temp.empty:
                        regr = linear_model.LinearRegression()
                        try:
                            regr.fit(np.matrix(temp.ix[:,0]).T, np.matrix(temp.ix[:,1]).T)
                            tempRes.append(regr.coef_)
                        except ValueError:
                            continue
                    else:
                        continue
            try:
                lam = (tempRes[-1] - np.mean(tempRes[0:-1]))/np.std(tempRes[0:-1])
            except:
                continue
            t  = pd.DataFrame(lam, index = [stock])
            AllRes = pd.concat([AllRes,t])   
        return AllRes
    else:
        return ['002195.XSHE']
