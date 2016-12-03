def SmartMoney(stocks,date):  #We still need to furthur adjust this factor by buying stocks only when CSI is lower
    window = 1
    start = date
    AllRes = []
    indstocks = []
    for stock in list(stocks):
        TradingDetails = get_price(stock, start_date=end, end_date=end, frequency='1m')[['OpeningPx','TotalVolumeTraded']]
        TradingDetails['PChange'] = np.log(TradingDetails.OpeningPx/TradingDetails.OpeningPx.shift(1))
        TradingDetails['Smart'] = abs(TradingDetails['PChange'])/np.sqrt(TradingDetails.TotalVolumeTraded)
        TradingDetails = TradingDetails.sort_values(['Smart'],ascending=False)
        SmartPart = TradingDetails.iloc[0:int(0.2*TradingDetails.shape[0])]
        DumbPart = TradingDetails.iloc[int(0.2*TradingDetails.shape[0]):int(TradingDetails.shape[0])]
        try:
            SP = np.average(SmartPart.OpeningPx, weights=SmartPart.TotalVolumeTraded)
            DP = np.average(DumbPart.OpeningPx, weights=DumbPart.TotalVolumeTraded)
        except:
            continue
        Q = SP/DP
        #print(Q)
        AllRes.append(Q)
        indstocks.append(stock)
    AllRespd = pd.DataFrame(AllRes,index = indstocks)
    return AllRespd
