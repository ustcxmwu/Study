#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.
from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.faker import Faker

cq_city = ["北碚区", "巴南区", "渝北区", "九龙坡区","渝中区","江北区","南岸区","沙坪坝区","大渡口区"]
GDP_value = [552, 781, 1543, 1211,1204,1028,725,936,228]

def map_cq():
    c = (
        Map()
        .add("", [list(z) for z in zip(cq_city, GDP_value)], "重庆",is_map_symbol_show=False,)
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))

        .set_global_opts(
            title_opts=opts.TitleOpts(title="2018年重庆主城九区"),
            visualmap_opts=opts.VisualMapOpts(max_=99999,is_piecewise=True,
                            pieces=[{"max": 499, "min": 0, "label": "0-499","color":"#FFE4E1"},
                                    {"max": 899, "min": 500, "label": "500-899","color":"#FF7F50"},
                                    {"max": 1299, "min": 900, "label": "900-1299","color":"#F08080"},
                                    {"max": 1599, "min": 1300, "label": ">=1300","color":"#CD5C5C"}])
        )
    )
    return c


if __name__ == '__main__':
    cq = map_cq()
    cq.render(path="test_map_1.html")
