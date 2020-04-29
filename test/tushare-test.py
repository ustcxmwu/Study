import pandas as pd
import tushare as ts

def formatDate(Date, formatType='YYYYMMDD'):
    formatType = formatType.replace('YYYY', Date[0:4])
    formatType = formatType.replace('MM', Date[4:6])
    formatType = formatType.replace('DD', Date[-2:])
    return formatType


if __name__ == '__main__':
    # dataFrames = ts.get_stock_basics()
    # Code = dataFrames.index
    # # print(Code)
    #
    # code = '002356'
    # date = dataFrames.loc[code]['timeToMarket']  # 上市日期YYYYMMDD
    # date = formatDate(str(date), 'YYYY-MM-DD')  # 改一下格式
    # # 取600000的前复权所有日k线数据，取后复权数据autype='hfq'
    # dayKLin = ts.get_k_data(code=code, ktype='D', autype='qfq', start=date)
    # print(dayKLin)

    print('a' is 'a')
