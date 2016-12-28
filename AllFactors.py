#在做所有的计算之前都需要先把极端值去掉
def DealWithExtrem(FactorValue): #Here this function deal with the extrem values in FactorValue Series
    if isinstance(FactorValue, pd.DataFrame):
        FactorValue = FactorValue.iloc[0]
    #print(FactorValue)
    FactorValue = FactorValue.replace([np.inf, -np.inf], np.nan)
    FactorValue = FactorValue.dropna()
    #print('Before Extrem',FactorValue)
    median = FactorValue.median()
    mad = FactorValue.mad()
    FactorValue[FactorValue>(median+3*mad)] = (median+3*mad)
    FactorValue[FactorValue<(median-3*mad)] = (median-3*mad)
    #print('After Extrem',FactorValue)
    return FactorValue
#注意我们这里对于所有的因子都要进行标准化处理
#14-Day Relative Strength Index
def RSIIndividual(stock,end):
    window_length = 14
    start = "{:%Y-%m-%d}".format(datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=window_length))
    data = get_price(list(stock), start, end_date=end, frequency='1d', fields=None)['OpeningPx']
    close = data
    delta = close.diff()
    delta = delta[1:]
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = pd.stats.moments.ewma(up, window_length)
    roll_down1 = pd.stats.moments.ewma(down.abs(), window_length)
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))
    FactorValue = RSI1.iloc[-1]
    FactorValue = DealWithExtrem(FactorValue)
    FactorValue = (FactorValue - np.mean(FactorValue))/np.std(FactorValue)
    return FactorValue
#1M Price High - 1M Price Low 
def HighLow(stock,enddate):
    end = enddate
    startdate = "{:%Y-%m-%d}".format(datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=30))
    if len(stock) != 0 :
        prices = get_price(list(stock), startdate, end_date=enddate, frequency='1d', fields=None)['OpeningPx']
        ratio = (prices.max(axis = 0) - prices.iloc[-1])/abs((prices.min(axis = 0) - prices.iloc[-1]))
        FactorValue = ratio
        FactorValue = DealWithExtrem(FactorValue)
        return ratio
#1M Price Reversal,check again later 
def OneMonthsPriceReversal(stock,enddate):
    startdate = "{:%Y-%m-%d}".format(datetime.datetime.strptime(enddate, '%Y-%m-%d') - datetime.timedelta(days=30))
    if len(stock) != 0 :
        prices = get_price(list(stock), startdate, end_date=enddate, frequency='1d', fields=None)['OpeningPx']
        OneMonthsPriceReversal = pd.Series()
        for i in list(range(1,len(stock)+1)):
            OneMonthsPriceReversal = OneMonthsPriceReversal.append(pd.Series((prices.iloc[np.shape(prices)[0]-1][[stock[i-1]]]/prices.iloc[0][[stock[i-1]]])))
        returnv = (OneMonthsPriceReversal)
        FactorValue = returnv
        FactorValue = DealWithExtrem(FactorValue)
        FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
        return FactorValue

#24M Residual Return Variance 
def CAPMResidualVar(stock,enddate):
    end = enddate
    startdate = "{:%Y-%m-%d}".format(datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=1000))
    if len(stock) != 0 :
        indexprice = get_price('000016.XSHG', startdate, end_date=enddate, frequency='1d', fields=None, adjusted=True)['OpeningPx']
        prices = get_price(list(stock), startdate, end_date=enddate, frequency='1d', fields=None, adjusted=True)['OpeningPx']
        returns = np.log(prices/prices.shift(1)).iloc[1:-1]
        indexreturn = np.log(indexprice/indexprice.shift(1)).iloc[1:-1]
        ResAll = pd.Series()
        for i in list(range(1,len(stock)+1)):
            tempAll = pd.concat([returns[stock[i-1]],indexreturn],axis = 1)
            tempAll.columns = ['s','i']
            tempAll.replace([np.inf, -np.inf], np.nan)
            tempAll = tempAll.dropna()
            size = tempAll.shape
            if size[0] > 30:
                results = smf.ols('s~i',tempAll).fit()
                residual = tempAll['s'] - results.predict()
                tempRes = residual.std()**2
            else:
                tempRes = np.nan
            ResAll = ResAll.set_value(stock[i-1], tempRes)
            FactorValue = ResAll
            FactorValue = DealWithExtrem(FactorValue)
            FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
        return FactorValue
    else:
        return 'Error'
#Ind Grp Rel Cash to EV Ratio 
def Cash2EV(stock,enddate):
    date = enddate
    fundamental_df = get_fundamentals(
        query(
            fundamentals.balance_sheet.cash, 
            fundamentals.eod_derivative_indicator.market_cap, 
            fundamentals.balance_sheet.total_liabilities, 
            fundamentals.balance_sheet.cash_equivalent,
            fundamentals.balance_sheet.current_investment
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        ),entry_date = enddate
    )
    #fundamental_df = fundamental_df.T
    Cash = fundamental_df['cash']
    EV = fundamental_df['market_cap'] + fundamental_df['total_liabilities'] - fundamental_df['cash']
    FactorValue = Cash/EV
    size = FactorValue.shape
    if isinstance(FactorValue, pd.DataFrame):
        FactorValue = FactorValue.iloc[0]
    FactorValue = DealWithExtrem(FactorValue)
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue
def IndCash2EV(stock,enddate):
    AllResult = Cash2EV(stock,enddate)
    t1 = AllResult #For later reference purpose
    allind = ['A01','A02','A03','A04','A05','B06','B07','B08','B09','B10','B11','B12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31','C32','C33','C34','C35','C36','C37','C38','C39','C40','C41','C42','C43','D44','D45','D46','E47','E48','E49','E50','F51','F52','G53','G54','G55','G56','G57','G58','G59','G60','H61','H62','I63','I64','I65','J66','J67','J68','J69','K70','L71','L72','M73','M74','M75','N76','N77','N78','O79','O80','O81','P82','Q83','Q84','R85','R86','R87','R88','R89']
    StockTotal = pd.Series()
    for ind in allind:
        stocks = industry(ind)
        #print(len(stocks))
        indRes = AllResult[stocks]
        indMean = indRes.mean()
        indStd = indRes.std()
        StocksResult = (indRes - indMean)/indStd
        StockTotal = StockTotal.append(StocksResult) 
    a = StockTotal.dropna() #Delete all NAs
    a = a[~a.index.duplicated()] #Only keep the unique indices
    concatm = pd.concat([t1,a],axis = 1) #Make sure our return length is 300
    concatm.columns = ['reference','factor']
    FactorValue = concatm['factor']
    FactorValue = DealWithExtrem(FactorValue)
    return FactorValue
#test = IndCash2EV(stock,enddate)
#Ind Grp Rel Cash to Price 
def Cash2Price(stock,enddate):
    date = enddate
    fundamental_df = get_fundamentals(
        query(
            fundamentals.balance_sheet.cash, 
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        ),entry_date = enddate
    )
    #fundamental_df = fundamental_df.T
    Cash = fundamental_df['cash']
    startdate = "{:%Y-%m-%d}".format(datetime.datetime.strptime(enddate, '%Y-%m-%d') - datetime.timedelta(days=10))
    if len(stock) != 0 :
        prices = get_price(list(stock), startdate, end_date=enddate, frequency='1d', fields=None)['OpeningPx']
    FactorValue = Cash/prices.iloc[-1]
    size = FactorValue.shape
    FactorValue = DealWithExtrem(FactorValue)
    if isinstance(FactorValue, pd.DataFrame):
        FactorValue = FactorValue.iloc[0]
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue
def IndCash2Price(stock,enddate):
    AllResult = Cash2Price(stock,enddate)
    t1 = AllResult #For later reference purpose
    allind = ['A01','A02','A03','A04','A05','B06','B07','B08','B09','B10','B11','B12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31','C32','C33','C34','C35','C36','C37','C38','C39','C40','C41','C42','C43','D44','D45','D46','E47','E48','E49','E50','F51','F52','G53','G54','G55','G56','G57','G58','G59','G60','H61','H62','I63','I64','I65','J66','J67','J68','J69','K70','L71','L72','M73','M74','M75','N76','N77','N78','O79','O80','O81','P82','Q83','Q84','R85','R86','R87','R88','R89']
    StockTotal = pd.Series()
    for ind in allind:
        stocks = industry(ind)
        #print(len(stocks))
        indRes = AllResult[stocks]
        indMean = indRes.mean()
        indStd = indRes.std()
        StocksResult = (indRes - indMean)/indStd
        StockTotal = StockTotal.append(StocksResult) 
    a = StockTotal.dropna() #Delete all NAs
    a = a[~a.index.duplicated()] #Only keep the unique indices
    concatm = pd.concat([t1,a],axis = 1) #Make sure our return length is 300
    concatm.columns = ['reference','factor']
    FactorValue = concatm['factor']
    FactorValue = DealWithExtrem(FactorValue)
    return FactorValue
#test = IndCash2Price(stock,enddate)
#test1 = IndCash2EV(stock,enddate)
#Ind Grp Rel Cash to Total Assets  
def Cash2TotalAsset(stock,enddate):
    date = enddate
    fundamental_df = get_fundamentals(
        query(
            fundamentals.balance_sheet.cash, 
            fundamentals.balance_sheet.total_assets
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        ),entry_date = enddate
    )
    #fundamental_df = fundamental_df.T
    Cash = fundamental_df['cash']
    TotalAsset = fundamental_df['total_assets']
    FactorValue = Cash/TotalAsset
    size = FactorValue.shape
    if isinstance(FactorValue, pd.DataFrame):
        FactorValue = FactorValue.iloc[0]
    FactorValue = DealWithExtrem(FactorValue)
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue
def IndCash2TotalAsset(stock,enddate):
    AllResult = Cash2TotalAsset(stock,enddate)
    t1 = AllResult #For later reference purpose
    allind = ['A01','A02','A03','A04','A05','B06','B07','B08','B09','B10','B11','B12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31','C32','C33','C34','C35','C36','C37','C38','C39','C40','C41','C42','C43','D44','D45','D46','E47','E48','E49','E50','F51','F52','G53','G54','G55','G56','G57','G58','G59','G60','H61','H62','I63','I64','I65','J66','J67','J68','J69','K70','L71','L72','M73','M74','M75','N76','N77','N78','O79','O80','O81','P82','Q83','Q84','R85','R86','R87','R88','R89']
    StockTotal = pd.Series()
    for ind in allind:
        stocks = industry(ind)
        #print(len(stocks))
        indRes = AllResult[stocks]
        indMean = indRes.mean()
        indStd = indRes.std()
        StocksResult = (indRes - indMean)/indStd
        StockTotal = StockTotal.append(StocksResult) 
    a = StockTotal.dropna() #Delete all NAs
    a = a[~a.index.duplicated()] #Only keep the unique indices
    concatm = pd.concat([t1,a],axis = 1) #Make sure our return length is 300
    concatm.columns = ['reference','factor']
    FactorValue = concatm['factor']
    FactorValue = DealWithExtrem(FactorValue)
    return FactorValue
#IndCash2TotalAsset(stock,enddate)
#Ind Grp Rel EBITDA to Price 
def EBITDA2Price(stock,enddate):
    fundamental_df = fundamental_df = get_fundamentals(
        query(
            fundamentals.financial_indicator.ebit
        ).filter(
            fundamentals.financial_indicator.stockcode.in_(stock)
        ),enddate
    )
    #fundamental_df = fundamental_df.T #Seems we only need to use this line in the backtesting process
    prices = get_price(list(stock), "{:%Y-%m-%d}".format(datetime.datetime.strptime(enddate, '%Y-%m-%d') - datetime.timedelta(days=10)), enddate, frequency='1d', fields=None)['OpeningPx']
    FactorValue = fundamental_df['ebit']/prices.iloc[-1]
    FactorValue = DealWithExtrem(FactorValue)
    size = FactorValue.shape
    if isinstance(FactorValue, pd.DataFrame):
        FactorValue = FactorValue.iloc[0]
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue
def IndEBITDA2Price(stock,enddate):
    AllResult = EBITDA2Price(stock,enddate)
    t1 = AllResult #For later reference purpose
    allind = ['A01','A02','A03','A04','A05','B06','B07','B08','B09','B10','B11','B12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31','C32','C33','C34','C35','C36','C37','C38','C39','C40','C41','C42','C43','D44','D45','D46','E47','E48','E49','E50','F51','F52','G53','G54','G55','G56','G57','G58','G59','G60','H61','H62','I63','I64','I65','J66','J67','J68','J69','K70','L71','L72','M73','M74','M75','N76','N77','N78','O79','O80','O81','P82','Q83','Q84','R85','R86','R87','R88','R89']
    StockTotal = pd.Series()
    for ind in allind:
        stocks = industry(ind)
        #print(len(stocks))
        indRes = AllResult[stocks]
        indMean = indRes.mean()
        indStd = indRes.std()
        StocksResult = (indRes - indMean)/indStd
        StockTotal = StockTotal.append(StocksResult) 
    a = StockTotal.dropna() #Delete all NAs
    a = a[~a.index.duplicated()] #Only keep the unique indices
    concatm = pd.concat([t1,a],axis = 1) #Make sure our return length is 300
    concatm.columns = ['reference','factor']
    FactorValue = concatm['factor']
    FactorValue = DealWithExtrem(FactorValue)
    return FactorValue

#Ind Grp Rel Inverse P/E Ratio Adj for Growth and Yield


#3Y Chg in Sales to Price,可以用operating_revenue来代替 
def Sale2PriceChange(stock,enddate):
    fundamental_df = fundamental_df = get_fundamentals(
        query(
            fundamentals.cash_flow_statement.cash_received_from_sales_of_goods,
            fundamentals.eod_derivative_indicator.market_cap
        ).filter(
            fundamentals.financial_indicator.stockcode.in_(stock)
        ),enddate,interval = '3y'
    )
    #fundamental_df = fundamental_df.T
    r1 = fundamental_df['cash_received_from_sales_of_goods'].iloc[0]/fundamental_df['market_cap'].iloc[0]
    r2 = fundamental_df['cash_received_from_sales_of_goods'].iloc[1]/fundamental_df['market_cap'].iloc[0]
    r3 = fundamental_df['cash_received_from_sales_of_goods'].iloc[2]/fundamental_df['market_cap'].iloc[0]
    All = pd.concat([r1,r2,r3],axis = 1)
    FactorValue  = All.T.iloc[0]/All.T.iloc[-1]
    FactorValue = DealWithExtrem(FactorValue)
    size = FactorValue.shape
    if isinstance(FactorValue, pd.DataFrame):
        FactorValue = FactorValue.iloc[0]
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue
#FactorValue = Sale2PriceChange(stock,enddate)
#5 Yr Hist Rel Assets to Price Ratio,assets 可以用财务数据里的total_assets来代替

#Ind Grp Rel 1Y Chg in Operating Cash Flow per Share  
def NetCashFlow2EV(stock,enddate):
    date = enddate
    fundamental_df = get_fundamentals(
        query(
            fundamentals.financial_indicator.operating_cash_flow_per_share
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        ),entry_date = enddate,interval = '3y'
    )
    #fundamental_df = fundamental_df.T
    Cash = fundamental_df['free_cash_flow_company_per_share']
    EV = fundamental_df['market_cap'] + fundamental_df['total_liabilities'] - fundamental_df['cash']
    FactorValue = Cash/EV
    FactorValue = DealWithExtrem(FactorValue)
    size = FactorValue.shape
    if isinstance(FactorValue, pd.DataFrame):
        FactorValue = FactorValue.iloc[0]
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue
def IndNetCashFlow2EV(stock,enddate):
    AllResult = NetCashFlow2EV(stock,enddate)
    t1 = AllResult #For later reference purpose
    allind = ['A01','A02','A03','A04','A05','B06','B07','B08','B09','B10','B11','B12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31','C32','C33','C34','C35','C36','C37','C38','C39','C40','C41','C42','C43','D44','D45','D46','E47','E48','E49','E50','F51','F52','G53','G54','G55','G56','G57','G58','G59','G60','H61','H62','I63','I64','I65','J66','J67','J68','J69','K70','L71','L72','M73','M74','M75','N76','N77','N78','O79','O80','O81','P82','Q83','Q84','R85','R86','R87','R88','R89']
    StockTotal = pd.Series()
    for ind in allind:
        stocks = industry(ind)
        #print(len(stocks))
        indRes = AllResult[stocks]
        indMean = indRes.mean()
        indStd = indRes.std()
        StocksResult = (indRes - indMean)/indStd
        StockTotal = StockTotal.append(StocksResult) 
    a = StockTotal.dropna() #Delete all NAs
    a = a[~a.index.duplicated()] #Only keep the unique indices
    concatm = pd.concat([t1,a],axis = 1) #Make sure our return length is 300
    concatm.columns = ['reference','factor']
    FactorValue = concatm['factor']
    FactorValue = DealWithExtrem(FactorValue)
    return FactorValue

#Ind Grp Rel Net Cash Flow to Enterprise Value
def NetCashFlow2EV(stock,enddate): 
    date = enddate
    fundamental_df = get_fundamentals(
        query(
            fundamentals.balance_sheet.cash, 
            fundamentals.eod_derivative_indicator.market_cap, 
            fundamentals.balance_sheet.total_liabilities, 
            fundamentals.balance_sheet.cash_equivalent,
            fundamentals.financial_indicator.free_cash_flow_company_per_share
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        ),entry_date = enddate
    )
    #fundamental_df = fundamental_df.T
    Cash = fundamental_df['free_cash_flow_company_per_share']
    EV = fundamental_df['market_cap'] + fundamental_df['total_liabilities'] - fundamental_df['cash']
    FactorValue = Cash/EV
    FactorValue = DealWithExtrem(FactorValue)
    size = FactorValue.shape
    if isinstance(FactorValue, pd.DataFrame):
        FactorValue = FactorValue.iloc[0]
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue
def IndNetCashFlow2EV(stock,enddate):
    AllResult = NetCashFlow2EV(stock,enddate)
    t1 = AllResult #For later reference purpose
    allind = ['A01','A02','A03','A04','A05','B06','B07','B08','B09','B10','B11','B12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31','C32','C33','C34','C35','C36','C37','C38','C39','C40','C41','C42','C43','D44','D45','D46','E47','E48','E49','E50','F51','F52','G53','G54','G55','G56','G57','G58','G59','G60','H61','H62','I63','I64','I65','J66','J67','J68','J69','K70','L71','L72','M73','M74','M75','N76','N77','N78','O79','O80','O81','P82','Q83','Q84','R85','R86','R87','R88','R89']
    StockTotal = pd.Series()
    for ind in allind:
        stocks = industry(ind)
        #print(len(stocks))
        indRes = AllResult[stocks]
        indMean = indRes.mean()
        indStd = indRes.std()
        StocksResult = (indRes - indMean)/indStd
        StockTotal = StockTotal.append(StocksResult) 
    a = StockTotal.dropna() #Delete all NAs
    a = a[~a.index.duplicated()] #Only keep the unique indices
    concatm = pd.concat([t1,a],axis = 1) #Make sure our return length is 300
    concatm.columns = ['reference','factor']
    FactorValue = concatm['factor']
    FactorValue = DealWithExtrem(FactorValue)
    return FactorValue
#IndNetCashFlow2EV(stock,enddate)
#Ind Grp Rel Operating Cash Flow to Price 
def EquityOCFP(stock,enddate):
    date = enddate
    fundamental_df = get_fundamentals(
        query(
            fundamentals.financial_indicator.operating_cash_flow_per_share 
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        ),entry_date = enddate
    )
    prices = get_price(list(stock), "{:%Y-%m-%d}".format(datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=10)), date, frequency='1d', fields=None)['OpeningPx']
    #fundamental_df = fundamental_df.T
    FactorValue = fundamental_df['operating_cash_flow_per_share']/prices.iloc[-1]
    FactorValue = DealWithExtrem(FactorValue)
    size = FactorValue.shape
    if isinstance(FactorValue, pd.DataFrame):
        FactorValue = FactorValue.iloc[0]
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue

def indOCFP(stock,enddate):
    AllResult = EquityOCFP(stock,enddate)
    t1 = AllResult #For later reference purpose
    allind = ['A01','A02','A03','A04','A05','B06','B07','B08','B09','B10','B11','B12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31','C32','C33','C34','C35','C36','C37','C38','C39','C40','C41','C42','C43','D44','D45','D46','E47','E48','E49','E50','F51','F52','G53','G54','G55','G56','G57','G58','G59','G60','H61','H62','I63','I64','I65','J66','J67','J68','J69','K70','L71','L72','M73','M74','M75','N76','N77','N78','O79','O80','O81','P82','Q83','Q84','R85','R86','R87','R88','R89']
    StockTotal = pd.Series()
    for ind in allind:
        stocks = industry(ind)
        indRes = AllResult[stocks]
        indMean = indRes.mean()
        indStd = indRes.std()
        StocksResult = (indRes - indMean)/indStd
        StockTotal = StockTotal.append(StocksResult) 
    a = StockTotal.dropna() #Delete all NAs
    a = a[~a.index.duplicated()] #Only keep the unique indices
    concatm = pd.concat([t1,a],axis = 1) #Make sure our return length is 300
    concatm.columns = ['reference','factor']
    FactorValue = concatm['factor']
    FactorValue = DealWithExtrem(FactorValue)
    return FactorValue
#5 Yr Hist Rel Inverse P/E Ratio Adj for Growth and Yield

#5 Yr Hist Rel Net Cash Flow to Enterprise Value

#5 Yr Hist Rel Operating Cash Flow to Price

#Ind Grp Rel Inverse P/E Ratio Adj for Growth and Yield

#5 Yr Hist Rel Assets to Price Ratio,assets 可以用财务数据里的total_assets来代替

#Inverse P/E Ratio Adj for Growth and Yield 
def EPAdjGrowth(stock,enddate):
    fundamental_df = fundamental_df = get_fundamentals(
        query(
            fundamentals.financial_indicator.earnings_per_share
        ).filter(
            fundamentals.financial_indicator.stockcode.in_(stock)
        ),enddate
    )
    #fundamental_df = fundamental_df.T #Seems we only need to use this line in the backtesting process
    prices = get_price(list(stock), "{:%Y-%m-%d}".format(datetime.datetime.strptime(enddate, '%Y-%m-%d') - datetime.timedelta(days=10)), enddate, frequency='1d', fields=None)['OpeningPx']
    EPRatio = fundamental_df['earnings_per_share']/prices.iloc[-1]
    size = FactorValue.shape
    FactorValue = DealWithExtrem(FactorValue)
    if isinstance(FactorValue, pd.DataFrame):
        FactorValue = FactorValue.iloc[0]
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue

#60M CAPM Alpha
def CAPMAlpha60M(stock,enddate):
    end = enddate
    startdate = "{:%Y-%m-%d}".format(datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=1000))
    if len(stock) != 0 :
        indexprice = get_price('000016.XSHG', startdate, end_date=enddate, frequency='1d', fields=None, adjusted=True)['OpeningPx']
        prices = get_price(list(stock), startdate, end_date=enddate, frequency='1d', fields=None, adjusted=True)['OpeningPx']
        returns = np.log(prices/prices.shift(1)).iloc[1:-1]
        indexreturn = np.log(indexprice/indexprice.shift(1)).iloc[1:-1]
        Beta = pd.Series()
        Alpha = pd.Series()
        tValue = pd.Series()
        for i in list(range(1,len(stock)+1)):
            tempAll = pd.concat([returns[stock[i-1]],indexreturn],axis = 1)
            tempAll.columns = ['s','i']
            tempAll = tempAll.replace([np.inf, -np.inf], np.nan)
            tempAll = tempAll.dropna()
            size = tempAll.shape
            if size[0] > 30:
                results = smf.ols('s~i',tempAll).fit()
                alpha = results.params[0]
                beta = results.params[1]
                tv = results.tvalues[0]
            else:
                alpha = np.nan
                beta = np.nan    
                tv = np.nan 
            Beta = Beta.set_value(stock[i-1], beta)
            Alpha = Alpha.set_value(stock[i-1], alpha)
            tValue = tValue.set_value(stock[i-1], tv)
            FactorValue = Alpha
            FactorValue = DealWithExtrem(FactorValue)
        return FactorValue
    else:
        return 'Error'

#60M t value of CAPM Alpha, this mean the corresponding stock is consistently outperforming the market
def CAPMAlphat60M(stock,enddate):
    end = enddate
    startdate = "{:%Y-%m-%d}".format(datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=1000))
    if len(stock) != 0 :
        indexprice = get_price('000016.XSHG', startdate, end_date=enddate, frequency='1d', fields=None, adjusted=True)['OpeningPx']
        prices = get_price(list(stock), startdate, end_date=enddate, frequency='1d', fields=None, adjusted=True)['OpeningPx']
        returns = np.log(prices/prices.shift(1)).iloc[1:-1]
        indexreturn = np.log(indexprice/indexprice.shift(1)).iloc[1:-1]
        Beta = pd.Series()
        Alpha = pd.Series()
        tValue = pd.Series()
        for i in list(range(1,len(stock)+1)):
            tempAll = pd.concat([returns[stock[i-1]],indexreturn],axis = 1)
            tempAll.columns = ['s','i']
            tempAll = tempAll.replace([np.inf, -np.inf], np.nan)
            tempAll = tempAll.dropna()
            size = tempAll.shape
            if size[0] > 30:
                results = smf.ols('s~i',tempAll).fit()
                alpha = results.params[0]
                beta = results.params[1]
                tv = results.tvalues[0]
            else:
                alpha = np.nan
                beta = np.nan    
                tv = np.nan 
            Beta = Beta.set_value(stock[i-1], beta)
            Alpha = Alpha.set_value(stock[i-1], alpha)
            tValue = tValue.set_value(stock[i-1], tv)
            FactorValue = tValue
            FactorValue = DealWithExtrem(FactorValue)
        return FactorValue
    else:
        return 'Error'

#6M Momentum in Trailing 12M Sales, 我们可以用operating revenue来代替

#Closing Price to 260 Day Low
def PricetoLowest(stock,end):
    start = "{:%Y-%m-%d}".format(datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=260))
    if len(stock) != 0 :
        prices = get_price(list(stock), start, end, frequency='1d', fields=None)['OpeningPx']
        RatiotoLowest = pd.Series()
        for i in list(range(1,len(stock)+1)):
            RatiotoLowest = RatiotoLowest.append(pd.Series((prices.iloc[np.shape(prices)[0]-1][[stock[i-1]]]/min(prices[stock[i-1]]))))
        returnv = (RatiotoLowest)
        FactorValue = returnv
        FactorValue = DealWithExtrem(FactorValue)
        FactorValue = (FactorValue - np.mean(FactorValue))/np.std(FactorValue)
        return FactorValue
    else:
        return 'Error'
#Dividends to Price Ratio 
def DDPRatio(stock,enddate):
    fundamental_df = fundamental_df = get_fundamentals(
        query(
            fundamentals.financial_indicator.dividend_per_share
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        ),enddate
    )
    #fundamental_df = fundamental_df.T #Seems we only need to use this line in the backtesting process
    prices = get_price(list(stock), "{:%Y-%m-%d}".format(datetime.datetime.strptime(enddate, '%Y-%m-%d') - datetime.timedelta(days=10)), enddate, frequency='1d', fields=None)['OpeningPx']
    FactorValue = fundamental_df['dividend_per_share']/prices.iloc[-1]
    size = FactorValue.shape
    FactorValue = DealWithExtrem(FactorValue)
    if isinstance(FactorValue, pd.DataFrame):
        FactorValue = FactorValue.iloc[0]
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue

#Earnings to Price  
def EPRatio(stock,enddate):
    fundamental_df = fundamental_df = get_fundamentals(
        query(
            fundamentals.financial_indicator.earnings_per_share
        ).filter(
            fundamentals.financial_indicator.stockcode.in_(stock)
        ),enddate
    )
    #fundamental_df = fundamental_df.T #Seems we only need to use this line in the backtesting process
    FactorValue = fundamental_df['earnings_per_share']
    prices = get_price(list(stock), "{:%Y-%m-%d}".format(datetime.datetime.strptime(enddate, '%Y-%m-%d') - datetime.timedelta(days=10)), enddate, frequency='1d', fields=None)['OpeningPx']
    FactorValue = fundamental_df['earnings_per_share']/prices.iloc[-1]
    size = FactorValue.shape
    FactorValue = DealWithExtrem(FactorValue)
    if isinstance(FactorValue, pd.DataFrame):
        FactorValue = FactorValue.iloc[0]
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue

#Ind Grp Rel 1M Price Reversal 
def OneMonthsPriceReversal(stock,enddate):
    startdate = "{:%Y-%m-%d}".format(datetime.datetime.strptime(enddate, '%Y-%m-%d') - datetime.timedelta(days=30))
    if len(stock) != 0 :
        prices = get_price(list(stock), startdate, end_date=enddate, frequency='1d', fields=None)['OpeningPx']
        OneMonthsPriceReversal = prices.iloc[-1]/prices.iloc[0]
        returnv = (OneMonthsPriceReversal)
        FactorValue = returnv
        FactorValue = DealWithExtrem(FactorValue)
        FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
        return FactorValue
def IndGrpOneMonthsPriceReversal(stock,enddate):
    AllResult = OneMonthsPriceReversal(stock,enddate)
    t1 = AllResult #For later reference purpose
    allind = ['A01','A02','A03','A04','A05','B06','B07','B08','B09','B10','B11','B12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31','C32','C33','C34','C35','C36','C37','C38','C39','C40','C41','C42','C43','D44','D45','D46','E47','E48','E49','E50','F51','F52','G53','G54','G55','G56','G57','G58','G59','G60','H61','H62','I63','I64','I65','J66','J67','J68','J69','K70','L71','L72','M73','M74','M75','N76','N77','N78','O79','O80','O81','P82','Q83','Q84','R85','R86','R87','R88','R89']
    StockTotal = pd.Series()
    for ind in allind:
        stocks = industry(ind)
        #print(len(stocks))
        indRes = AllResult[stocks]
        indMean = indRes.mean()
        indStd = indRes.std()
        StocksResult = (indRes - indMean)/indStd
        StockTotal = StockTotal.append(StocksResult) 
    a = StockTotal.dropna() #Delete all NAs
    a = a[~a.index.duplicated()] #Only keep the unique indices
    concatm = pd.concat([t1,a],axis = 1) #Make sure our return length is 300
    concatm.columns = ['reference','factor']
    FactorValue = concatm['factor']
    FactorValue = DealWithExtrem(FactorValue)
    return FactorValue

#Inverse PEG
def IPEG(stock,enddate):
    date = enddate
    fundamental_df = get_fundamentals(
        query(
            fundamentals.eod_derivative_indicator.peg_ratio 
        ).filter(
            fundamentals.eod_derivative_indicator.stockcode.in_(stock)
        ),entry_date = enddate
    )
    #fundamental_df = fundamental_df.T
    #print(fundamental_df)
    FactorValue = fundamental_df['peg_ratio']
    #print(FactorValue)
    FactorValue = DealWithExtrem(FactorValue)
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return 1/FactorValue

#Log Market Cap,不同的印子处理方式即使在标准化之后也依然是有影响的
def EquitySize(stock,enddate):
    date = enddate
    fundamental_df = get_fundamentals(
        query(
            fundamentals.eod_derivative_indicator.market_cap 
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        ),entry_date = enddate
    )
    #fundamental_df = fundamental_df.T
    FactorValue = fundamental_df['market_cap']
    FactorValue = DealWithExtrem(FactorValue)
    FactorValue = np.log(FactorValue)
    #FactorValue = FactorValue.iloc[0]
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue

#Cubbed Market Cap,不同的印子处理方式即使在标准化之后也依然是有影响的
def EquitySizeCubbed(stock,enddate):
    date = enddate
    fundamental_df = get_fundamentals(
        query(
            fundamentals.eod_derivative_indicator.market_cap 
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        ),entry_date = enddate
    )
    
    #fundamental_df = fundamental_df.T
    FactorValue = fundamental_df['market_cap']
    FactorValue = DealWithExtrem(FactorValue)
    FactorValue = FactorValue ** 3
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue

#Log of Unadjusted Stock Price
def UnadjustedPrice(stock,enddate):
    startdate = "{:%Y-%m-%d}".format(datetime.datetime.strptime(enddate, '%Y-%m-%d') - datetime.timedelta(days=1))
    if len(stock) != 0 :
        prices = get_price(list(stock), startdate, end_date=enddate, frequency='1d', fields=None)['OpeningPx']
        FactorValue = prices.iloc[-1]
        FactorValue = DealWithExtrem(FactorValue)
        return FactorValue

#Max Daily Return in the Past 6 Months
def MaxReturn6m(stock,enddate):
    startdate = "{:%Y-%m-%d}".format(datetime.datetime.strptime(enddate, '%Y-%m-%d') - datetime.timedelta(days=180))
    if len(stock) != 0 :
        prices = get_price(list(stock), startdate, end_date=enddate, frequency='1d', fields=None)['OpeningPx']
        ret = np.log(prices/prices.shift(1))
        ret = ret.iloc[1:]
        FactorValue = ret.max()
        FactorValue = DealWithExtrem(FactorValue)
        return FactorValue

#6 Months Price Reversal
def SixMonthsPriceReversal(stock,enddate):
    startdate = "{:%Y-%m-%d}".format(datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=180))
    if len(stock) != 0 :
        prices = get_price(list(stock), startdate, end_date=enddate, frequency='1d', fields=None)['OpeningPx']
        SixMonthsPriceReversal = prices.iloc[-1]/prices.iloc[0]
        FactorValue = SixMonthsPriceReversal
        FactorValue = DealWithExtrem(FactorValue)
        FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
        return FactorValue
    else:
        return np.nan
    
#Book to Price

#Ind Grp Rel Book to Price, book to price的计算方法根据excel里面查到的方法来做
def BPRatio(stock,enddate):
    fundamental_df = fundamental_df = get_fundamentals(
        query(
            fundamentals.financial_indicator.book_value_per_share
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        ),enddate
    )
    #fundamental_df = fundamental_df.T #Seems we only need to use this line in the backtesting process
    prices = get_price(list(stock), "{:%Y-%m-%d}".format(datetime.datetime.strptime(enddate, '%Y-%m-%d') - datetime.timedelta(days=10)), enddate, frequency='1d', fields=None)['OpeningPx']
    FactorValue = fundamental_df['book_value_per_share']/prices.iloc[-1]
    size = FactorValue.shape
    FactorValue = DealWithExtrem(FactorValue)
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue

def indBPRatio(stock,enddate):
    AllResult = BPRatio(stock,enddate)
    t1 = AllResult #For later reference purpose
    allind = ['A01','A02','A03','A04','A05','B06','B07','B08','B09','B10','B11','B12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31','C32','C33','C34','C35','C36','C37','C38','C39','C40','C41','C42','C43','D44','D45','D46','E47','E48','E49','E50','F51','F52','G53','G54','G55','G56','G57','G58','G59','G60','H61','H62','I63','I64','I65','J66','J67','J68','J69','K70','L71','L72','M73','M74','M75','N76','N77','N78','O79','O80','O81','P82','Q83','Q84','R85','R86','R87','R88','R89']
    StockTotal = pd.Series()
    for ind in allind:
        stocks = industry(ind)
        indRes = AllResult[stocks]
        indMean = indRes.mean()
        indStd = indRes.std()
        StocksResult = (indRes - indMean)/indStd
        StockTotal = StockTotal.append(StocksResult) 
    a = StockTotal.dropna() #Delete all NAs
    a = a[~a.index.duplicated()] #Only keep the unique indices
    concatm = pd.concat([t1,a],axis = 1) #Make sure our return length is 300
    concatm.columns = ['reference','factor']
    FactorValue = concatm['factor']
    FactorValue = DealWithExtrem(FactorValue)
    return FactorValue
#Ind Grp Rel Operating Cash Flow to Price
def EquityOCFP(stock,enddate):
    date = enddate
    fundamental_df = get_fundamentals(
        query(
            fundamentals.financial_indicator.operating_cash_flow_per_share 
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        ),entry_date = enddate
    )
    prices = get_price(list(stock), "{:%Y-%m-%d}".format(datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=10)), date, frequency='1d', fields=None)['OpeningPx']
    #fundamental_df = fundamental_df.T
    FactorValue = fundamental_df['operating_cash_flow_per_share']/prices.iloc[-1]
    FactorValue = DealWithExtrem(FactorValue)
    size = FactorValue.shape
    if isinstance(FactorValue, pd.DataFrame):
        FactorValue = FactorValue.iloc[0]
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue

def indOCFP(stock,enddate):
    AllResult = EquityOCFP(stock,enddate)
    t1 = AllResult #For later reference purpose
    allind = ['A01','A02','A03','A04','A05','B06','B07','B08','B09','B10','B11','B12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31','C32','C33','C34','C35','C36','C37','C38','C39','C40','C41','C42','C43','D44','D45','D46','E47','E48','E49','E50','F51','F52','G53','G54','G55','G56','G57','G58','G59','G60','H61','H62','I63','I64','I65','J66','J67','J68','J69','K70','L71','L72','M73','M74','M75','N76','N77','N78','O79','O80','O81','P82','Q83','Q84','R85','R86','R87','R88','R89']
    StockTotal = pd.Series()
    for ind in allind:
        stocks = industry(ind)
        indRes = AllResult[stocks]
        indMean = indRes.mean()
        indStd = indRes.std()
        StocksResult = (indRes - indMean)/indStd
        StockTotal = StockTotal.append(StocksResult) 
    a = StockTotal.dropna() #Delete all NAs
    a = a[~a.index.duplicated()] #Only keep the unique indices
    concatm = pd.concat([t1,a],axis = 1) #Make sure our return length is 300
    concatm.columns = ['reference','factor']
    FactorValue = concatm['factor']
    FactorValue = DealWithExtrem(FactorValue)
    return FactorValue

#Ind Grp Rel 6M Price Reversal，6 Months，很多都是Ind Grp这个概念
def SixMonthsPriceReversal(stock,enddate):
    startdate = "{:%Y-%m-%d}".format(datetime.datetime.strptime(enddate, '%Y-%m-%d') - datetime.timedelta(days=180))
    if len(stock) != 0 :
        prices = get_price(list(stock), startdate, end_date=enddate, frequency='1d', fields=None)['OpeningPx']
        OneMonthsPriceReversal = prices.iloc[-1]/prices.iloc[0]
        FactorValue = OneMonthsPriceReversal
        FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
        return FactorValue
def IndGrpSixMonthsPriceReversal(stock,enddate):
    AllResult = SixMonthsPriceReversal(stock,enddate)
    t1 = AllResult #For later reference purpose
    allind = ['A01','A02','A03','A04','A05','B06','B07','B08','B09','B10','B11','B12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31','C32','C33','C34','C35','C36','C37','C38','C39','C40','C41','C42','C43','D44','D45','D46','E47','E48','E49','E50','F51','F52','G53','G54','G55','G56','G57','G58','G59','G60','H61','H62','I63','I64','I65','J66','J67','J68','J69','K70','L71','L72','M73','M74','M75','N76','N77','N78','O79','O80','O81','P82','Q83','Q84','R85','R86','R87','R88','R89']
    StockTotal = pd.Series()
    for ind in allind:
        stocks = industry(ind)
        #print(len(stocks))
        indRes = AllResult[stocks]
        indMean = indRes.mean()
        indStd = indRes.std()
        StocksResult = (indRes - indMean)/indStd
        StockTotal = StockTotal.append(StocksResult) 
    a = StockTotal.dropna() #Delete all NAs
    a = a[~a.index.duplicated()] #Only keep the unique indices
    concatm = pd.concat([t1,a],axis = 1) #Make sure our return length is 300
    concatm.columns = ['reference','factor']
    FactorValue = concatm['factor']
    FactorValue = DealWithExtrem(FactorValue)
    return FactorValue

#Operating Cash Flow to Enterprise Value
def OCFPtoEV(stock,enddate):
    date = enddate
    fundamental_df = get_fundamentals(
        query(
            fundamentals.cash_flow_statement.cash_from_operating_activities, 
            fundamentals.eod_derivative_indicator.market_cap, 
            fundamentals.balance_sheet.total_liabilities, 
            fundamentals.balance_sheet.cash, 
            fundamentals.balance_sheet.cash_equivalent,
            fundamentals.balance_sheet.current_investment
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        ),entry_date = enddate
    )
    #fundamental_df = fundamental_df.T
    OCF = fundamental_df['cash_from_operating_activities']
    #print(OCF)
    EV = fundamental_df['market_cap'] + fundamental_df['total_liabilities'] - fundamental_df['cash']
    FactorValue = OCF/EV
    size = FactorValue.shape
    FactorValue = DealWithExtrem(FactorValue)
    if isinstance(FactorValue, pd.DataFrame):
        FactorValue = FactorValue.iloc[0]
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue

#Operating Cash Flow to Price 
def EquityOCFP(stock,enddate):
    date = enddate
    fundamental_df = get_fundamentals(
        query(
            fundamentals.financial_indicator.operating_cash_flow_per_share 
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        ),entry_date = enddate
    )
    prices = get_price(list(stock), "{:%Y-%m-%d}".format(datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=10)), date, frequency='1d', fields=None)['OpeningPx']
    #fundamental_df = fundamental_df.T
    FactorValue = fundamental_df['operating_cash_flow_per_share']/prices.iloc[-1]
    FactorValue = DealWithExtrem(FactorValue)
    size = FactorValue.shape
    if isinstance(FactorValue, pd.DataFrame):
        FactorValue = FactorValue.iloc[0]
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue

#Operating Earnings to Price Ratio
def EquityOCFP(stock,enddate):
    date = enddate
    fundamental_df = get_fundamentals(
        query(
            fundamentals.financial_indicator.operating_cash_flow_per_share 
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        ),entry_date = enddate
    )
    prices = get_price(list(stock), "{:%Y-%m-%d}".format(datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=10)), date, frequency='1d', fields=None)['OpeningPx']
    #fundamental_df = fundamental_df.T
    FactorValue = fundamental_df['operating_cash_flow_per_share']/prices.iloc[-1]
    size = FactorValue.shape
    FactorValue = DealWithExtrem(FactorValue)
    if isinstance(FactorValue, pd.DataFrame):
        FactorValue = FactorValue.iloc[0]
    FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
    return FactorValue

#Sharpe Ratio 
def Sharpe(stock,end):
    start = "{:%Y-%m-%d}".format(datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=360))
    if len(stock) != 0 :
        prices = get_price(list(stock), start, end, frequency='1d', fields=None)['OpeningPx']
        returns = pd.DataFrame(columns=(stock))
        for i in list(range(1,np.shape(prices)[0]-1)):
            returns.loc[i] = ((prices.iloc[i]/prices.iloc[i-1]-1))
        #print(returns)
        ave = np.nanmean(returns,axis = 0)
        #print(ave)
        std = np.nanstd(returns,axis = 0)
        #print(std)
        sharpe = ((ave-0.04)/std)*np.sqrt(252)
        #print(sharpe)
        returnv = pd.Series(sharpe,index = [stock])
        #print(returnv)
        FactorValue = returnv
        FactorValue = DealWithExtrem(FactorValue)
        FactorValue = FactorValue.replace([np.inf, -np.inf], np.nan)
        FactorValue = (FactorValue - FactorValue.mean())/FactorValue.std()
        return FactorValue
    else:
        return 'Error'
    
#Financial Leverage is also a good indicator 
def FLeverage(stock,enddate):
    fundamental_df = get_fundamentals(
        query(
            fundamentals.balance_sheet.long_term_liabilities, 
            fundamentals.balance_sheet.total_assets, 
            fundamentals.balance_sheet.total_equity,
            fundamentals.cash_flow_statement.cash_flow_from_operating_activities,
            fundamentals.financial_indicator.return_on_invested_capital,
            fundamentals.financial_indicator.return_on_equity,
            fundamentals.financial_indicator.return_on_asset
        ).filter(
            fundamentals.income_statement.stockcode.in_(stock)
        ),enddate
    ) 
    #fundamental_df = fundamental_df.T
    Asset = fundamental_df.total_assets
    TEquity = fundamental_df.total_equity
    Financial_Leverage = Asset/TEquity
    forCal = Financial_Leverage.replace([np.inf, -np.inf], np.nan)
    forCal = forCal.dropna()
    FactorValue = Financial_Leverage
    FactorValue = DealWithExtrem(FactorValue)
    FactorValue = (FactorValue - forCal.mean())/forCal.std()
    return FactorValue
    
    
def GetAllFactorExposure(stock,enddate,*args):  #Output is directly dataframe
    OutValue = []
    flag = 1
    for arg in args[0]:
        if flag == 1:
            OutValue = pd.DataFrame(arg(stock,enddate))
            #print(OutValue.shape)，这块基本上没问题了
            flag = 0
        else:
            OutValue = pd.concat([OutValue,arg(stock,enddate)],axis = 1)
            #print(OutValue.shape)
    OutValue.columns = list(range(1,len(args[0])+1))
    return OutValue
    

MyFactors = [IndEBITDA2Price,IndCash2TotalAsset ,IndCash2Price ,RSIIndividual, HighLow ,OneMonthsPriceReversal ,CAPMResidualVar ,IndCash2EV]
Part1 = GetAllFactorExposure(stock,enddate,MyFactors)
MyFactors = [CAPMAlphat60M,CAPMAlpha60M,indOCFP,IndNetCashFlow2EV,IndNetCashFlow2EV,IndCash2Price,IndCash2TotalAsset,IndEBITDA2Price,Sale2PriceChange]
Part2 = GetAllFactorExposure(stock,enddate,MyFactors)
MyFactors = [IndGrpSixMonthsPriceReversal,indOCFP,indBPRatio,BPRatio,SixMonthsPriceReversal,MaxReturn6m,IPEG,PricetoLowest, DDPRatio,EPRatio,IndGrpOneMonthsPriceReversal,EquitySize,EquitySizeCubbed,UnadjustedPrice]
Part3 = GetAllFactorExposure(stock,enddate,MyFactors)
MyFactors = [OCFPtoEV ,EquityOCFP,sharpe,FLeverage]
Part4 = GetAllFactorExposure(stock,enddate,MyFactors)
Part4
