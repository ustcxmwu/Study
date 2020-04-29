import requests
import re
import csv

def get_one_page(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text


def parse_one_page(html):
    pattern = re.compile('<dd>.*?board-index.*?>(\d+)</i>.*?src="(.*?)".*?name"><a.*?>(.*?)</a>.*?star">(.*?)</p>.*?releasetime">(.*?)</p.*?integer">(.*?)</i>.*?fraction">(.*?)</i>.*?</dd>', re.S)
    # re.S 表示匹配任意字符，如果不加，则无法匹配换行符
    items = re.findall(pattern, html)
    # print(items)
    for item in items:
        yield{
            'index': item[0],
        'thumb': get_thumb(item[1]), # 定义 get_thumb()方法进一步处理网址
        'name': item[2],
        'star': item[3].strip()[3:],
        # 'time': item[4].strip()[5:],
        # 用两个方法分别提取 time 里的日期和地区
        'time': get_release_time(item[4].strip()[5:]),
        'area': get_release_area(item[4].strip()[5:]),
        'score': item[5].strip() + item[6].strip()}


def get_thumb(url):
    pattern = re.compile(r'(.*?)@.*?')
    thumb = re.search(pattern, url)
    if thumb is not None:
        return thumb.group(1)


def get_release_time(data):
    pattern = re.compile(r'(.*?)(\(|$)')
    items = re.search(pattern, data)
    if items is None:
        return '未知'
    return items.group(1)  # 返回匹配到的第一个括号(.*?)中结果即时间


# http://p0.meituan.net/movie/5420be40e3b755ffe04779b9b199e935256906.jpg@160w_220h_1e_1c
# 去掉@160w_220h_1e_1c 就是大图
# 提取上映时间函数
def get_release_area(data):
    pattern = re.compile(r'.*\((.*)\)')
    # $表示匹配一行字符串的结尾，这里就是(.*?)；\(|$,表示匹配字符串含有(,或者只有(.*?)
    items = re.search(pattern, data)
    if items is None:
        return '未知'
    return items.group(1)


def write_to_csv(item):
    with open('猫眼 top100.csv', 'a', encoding='utf_8_sig',newline='') as f:
        # 'a'为追加模式（添加）
        # utf_8_sig 格式导出 csv 不乱码
        fieldnames = ['index', 'thumb', 'name', 'star', 'time', 'area', 'score']
        w = csv.DictWriter(f,fieldnames = fieldnames)
        # w.writeheader()
        w.writerow(item)


def main():
    url = 'http://maoyan.com/board/4?offset=0'
    html = get_one_page(url)
    for item in parse_one_page(html):
        print(item)
        write_to_csv(item)



if __name__ == '__main__':
    main()