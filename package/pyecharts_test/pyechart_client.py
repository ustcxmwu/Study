from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.faker import Faker


def fn1():
    c = (
        Map()
        .add("商家A", [list(z) for z in zip(Faker.guangdong_city, Faker.values())], "甘肃")
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Map-中国地图（带城市）"), visualmap_opts=opts.VisualMapOpts()
        )
        .render("map_guangdong.html")
    )


def shanxi():
    map2 = Map()
    city = ['太原市', '晋中市', '长治市', '临汾市', '运城市']
    values2 = [1.07, 3.85, 6.38, 8.21, 2.53]
    map2.add('山西aaa', [z for z in zip(city, values2)], '山西')
    map2.render("shanxi.html")




if __name__ == '__main__':
    # shanxi()
    fn1()