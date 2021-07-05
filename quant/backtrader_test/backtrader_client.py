from datetime import datetime
import backtrader as bt


class SmaCross(bt.SignalStrategy):
    def __init__(self):
        sma1, sma2 = bt.ind.SMA(period=10), bt.ind.SMA(period=30)
        crossover = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, crossover)


def main():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross)

    data0 = bt.feeds.YahooFinanceData(dataname='MSFT', fromdate=datetime(2011, 1, 1),
                                      todate=datetime(2011, 12, 31))
    cerebro.adddata(data0)

    cerebro.run()
    cerebro.plot()


if __name__ == "__main__":
    main()
