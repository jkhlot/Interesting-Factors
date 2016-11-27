import pandas as pd
import numpy as np
import datetime as datetime
from sklearn import datasets, linear_model

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
    

def Select(context,enddate):
    stocks = index_components('000300.XSHG')
    StockTotal = pd.Series()
    StocksResult = APMRaw(stocks,enddate)
    print(StocksResult[0:10])
    StockTotal = StocksResult.dropna()
    #StockTotal[StockTotal.index.values == '000001.XSHE']
    StockTotal = StockTotal.order(ascending=False)
    print(StockTotal[0:20])
    SelectedStocks = StockTotal.index.values[0:10]
    context.Stocks = SelectedStocks
    #logger.info("Interested at stock: " + str(context.Stocks))
    #update_universe(context.Stocks)

#Pleaseb be noted that we can not use some functions of bar_dict in before trading part, as some information are unknown at this time
def stoploss(context,bar_dict):
    for stock in context.Stocks:
        #print(stock)
        #print(bar_dict[stock].total_turnover)
        if bar_dict[stock].last<context.portfolio.positions[stock].average_cost*context.stoplossmultipler:
            order_target_percent(stock,0)
    pass 
    

def init(context):
    context.Stocks = '000300.XSHG'
    context.stoplossmultipler= 0.9  
    context.stoppofitmultipler= 1000.8 
    context.countdate = 0
    context.Traded = 0
    #scheduler.run_monthly(Select, tradingday=1)
    

def before_trading(context, bar_dict):
    context.countdate = context.countdate + 1
    if context.countdate%20 == 1:
        print("Here")
        enddate = "{:%Y-%m-%d}".format(get_previous_trading_date(context.now))
        print(enddate)
        Select(context,enddate)
        context.Stocks = context.Stocks[0:10]
        update_universe(context.Stocks)
        


def handle_bar(context, bar_dict):
    if context.Traded == 1:
        stoploss(context,bar_dict)
    if context.countdate%20 == 1:
        context.average_percent = 0.99/len(context.Stocks)
        #print(context.Stocks[0:5])
        for stock in context.Stocks:
            order_target_percent(stock, context.average_percent)
        context.Traded = 1
            #logger.info("Bought: " + str(context.average_percent) + " % for stock: " + str(stock))
            #context.fired = True
