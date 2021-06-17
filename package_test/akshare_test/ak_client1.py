#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import akshare as ak


if __name__ == '__main__':
    stock_zh_a_daily_hfq_df = ak.stock_zh_a_daily(symbol="sz000002", start_date='20201103', end_date='20201116',
                                                  adjust="hfq")
    print(stock_zh_a_daily_hfq_df)

    stock_em_hsgt_hold_stock_df = ak.stock_em_hsgt_hold_stock(market="北向", indicator="今日排行")
    print(stock_em_hsgt_hold_stock_df)