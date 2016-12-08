# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。
import pandas as pd
import numpy as np
from datetime import timedelta
from pybrain.datasets import SequentialDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.networks import Network
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer

# 训练trainX和trainY，并返回神经网络net
def train(context, trainX, trainY):
    try:
        ds = SequentialDataSet(4, 1)
        for dataX, dataY in zip(trainX, trainY):
            ds.addSample(dataX, dataY)
        net = buildNetwork(4, 5, 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True)
        trainer = RPropMinusTrainer(net, dataset=ds)
        EPOCHS_PER_CYCLE = 5
        CYCLES = 10
        for i in range(CYCLES):
            trainer.trainEpochs(EPOCHS_PER_CYCLE)
        return net
    except:
        pass

# 更新数据集data
def load(context, ticker):
    start = (context.now + timedelta(weeks=-25)).strftime('%Y-%m-%d')
    end = (context.now + timedelta(days=-1)).strftime('%Y-%m-%d')
    data = get_price(ticker,
                     start_date=start,
                     end_date=end,
                     frequency='1d',
                     fields = ['ClosingPx',
                               'HighPx',
                               'LowPx',
                               'TotalVolumeTraded'])
    return data

# 建模，每三个月运行一次
def modelize(context, bar_dict):
    if context.every3months != 2:
        context.every3months += 1
        return 0
    print('-'*20)
    print('modelizing')
    context.data = []
    context.net = []
    context.list = []
    for ticker in context.portfolio.positions:
        order_target_percent(ticker, 0)
    templist = list(get_fundamentals(query(fundamentals.eod_derivative_indicator.market_cap)
                                    .order_by(fundamentals.eod_derivative_indicator.market_cap.asc())
                                    .limit(context.num*10+10)).columns)[10:]
    update_universe(templist)
    for ticker in templist:
        sb = instruments(ticker).symbol
        if ('ST' not in sb) and ('退' not in sb):
            context.list.append(ticker)
            if len(context.list) == 5:
                break
    print('final list:')
    for ticker in context.list:
        print(ticker, instruments(ticker).symbol)
        data = load(context, ticker)
        trainX = np.array(data.ix[:-1,:])
        trainY = np.array(data.ix[1:,0])
        context.data.append(data)
        context.net.append(train(context, trainX, trainY))
    context.pct = [0.05] * context.num
    context.i = 1
    context.every3months = 0
    print('finished.')
    print('net length =', len(context.net))
    print('data length =', len(context.data))

# 最后利用每月更新的模型，每天进行交易，预测涨幅超过a就买入，预测跌幅超过b则卖出
def trade(context,bar_dict):
    ohnowhytheyarestillhere = [ticker for ticker in context.portfolio.positions if ticker not in context.list]
    for ticker in ohnowhytheyarestillhere:
        order_target_percent(ticker, 0)
    # 哈哈哈哈哈哈哈这样子欣泰电气之类的就不能赖着不走了
    mkt = history(3, '1d', 'close')['000001.XSHG']
    print('market {}% compared to yesterday'.format(mkt[-1]/mkt[-2]*100))
    if (mkt[-1]/mkt[-2] < 0.97 and mkt[-2]/mkt[-3] < 0.97) or mkt[-1]/mkt[-2] < 0.95:
        # 连续两天大盘跌破3个点，或者大盘跌破5个点
        for ticker in context.portfolio.positions:
            order_target_percent(ticker, 0)
        return 0
    if context.i == 0: modelize(context, bar_dict)
    predict_close = []
    actual_open = []
    for i in range(context.num):
        try:
            predict_close.append(context.net[i].activate(context.data[i].ix[-1,:])[0])
        except:
            predict_close.append(history(1, '1d', 'close')[context.list[i]].values[0])
        actual_open.append(history(1, '1m', 'close')[context.list[i]].values[0])   # 当前价位
    r = [round(pc / ao - 1, 4) for pc, ao in zip(predict_close, actual_open)]
    print('-'*40)
    print('predicted return today: {}'.format('  '.join([str(i) for i in r])))
    a, b = 0.8, 0.6
    for i in range(context.num):
        if r[i] > a:
            context.pct[i] = min(context.pct[i] + 0.10, 0.50)
        elif r[i] < b:
            context.pct[i] = max(context.pct[i] - 0.20, 0.00)
    pct = sum([context.portfolio.positions[ticker].market_value for ticker in context.portfolio.positions])/(context.portfolio.market_value+context.portfolio.cash)
    tot_pct = max(sum(context.pct),1)
    context.pct = list(map(lambda x: x/tot_pct, context.pct))
    for i in range(context.num):
        order_target_percent(context.list[i], context.pct[i])
    print('stock positions today: {}'.format('  '.join([str(round(i,2)) for i in context.pct])))
    print('total position:',pct)
    plot('total position', pct * 100)

# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    context.num = 5
    context.list = []
    context.i = 0
    context.every3months = 2
    context.pct = [0.05] * context.num
    context.net = []
    context.data = []
    scheduler.run_monthly(modelize,1)
    scheduler.run_daily(trade, time_rule=market_open(minute=1))

# before_trading此函数会在每天交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass

# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    pass
