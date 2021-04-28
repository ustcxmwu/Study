#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import akshare as ak


if __name__ == '__main__':
    stock_zh_a_daily_hfq_df = ak.stock_zh_a_daily(symbol="sz000002", start_date='20201103', end_date='20201116',
                                                  adjust="hfq")
    print(stock_zh_a_daily_hfq_df)
