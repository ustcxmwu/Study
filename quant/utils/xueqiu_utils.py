import json

import browser_cookie3
import requests
import yaml
from easydict import EasyDict as edict
from requests import Response

from quant.utils.east_money_utils import get_stocks

url = "http://quote.eastmoney.com/zixuan/"


def to_eastmoney_code(code: str):
    if code >= '333333':
        # 上海
        return '1%24{}'.format(code)
    else:
        return '0%24{}'.format(code)


def parse_resp(resp: Response, key=None):
    if resp.status_code != 200:
        raise Exception('code:{},msg:{}'.format(resp.status_code, resp.content))
    result = resp.text
    try:
        js_obj = json.loads(resp.text)
    except ValueError as e:
        try:
            js_obj = json.loads(resp.text[result.index('(') + 1:result.index(')')])
        except ValueError as e:
            raise ValueError("Response Json Error:{}".format(resp.text))

    data = js_obj.get('data', None)
    if data and key:
        result_value = data.get(key)
    else:
        result_value = data
    return js_obj['state'], result_value


def load_config():
    headers = {"content-type": "application/json",
               "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/92.0.4515.107 Safari/537.36",
               "Referer": "https://xueqiu.com/"}
    cookies_jar = browser_cookie3.chrome()
    return headers, cookies_jar


def parse_cookies(cookie: str):
    cj = browser_cookie3.chrome()  # firefox可以替换为browser_cookie3.firefox()
    return cj


def get_groups():
    headers, cj = load_config()
    url = "https://stock.xueqiu.com/v5/stock/portfolio/list.json?system=true"
    resp = requests.get(url, headers=headers, cookies=cj)
    if resp.status_code == 200:
        text = json.loads(resp.text)["data"]
        return text["stocks"]


def get_group_stocks():
    headers, cj = load_config()
    url = "https://stock.xueqiu.com/v5/stock/portfolio/stock/list.json?size=1000&category=1&pid=1"
    resp = requests.get(url, headers=headers, cookies=cj)
    if resp.status_code == 200:
        text = json.loads(resp.text)["data"]
        return text["stocks"]


def add_groups(group_name: str):
    headers, cj = load_config()
    url = "https://stock.xueqiu.com/v5/stock/portfolio/create.json?category=1&pnames={}".format(group_name)
    data = {
        "category": 1,
        "pnames": group_name
    }
    resp = requests.post(url, headers=headers, cookies=cj)
    if resp.status_code == 200:
        text = json.loads(resp.text)["data"]
        return text


def delete_groups(group_name: str):
    groups = get_groups()
    for ginfo in groups:
        if ginfo["name"] == group_name:
            ids = ginfo["id"]
    headers, cj = load_config()
    url = "https://stock.xueqiu.com/v5/stock/portfolio/delete.json?pids={}".format(ids)

    resp = requests.post(url, headers=headers, cookies=cj)
    if resp.status_code == 200:
        pass


def add_stock_to_group(group_name: str, stock: str):
    headers, cj = load_config()
    url = "https://stock.xueqiu.com/v5/stock/portfolio/stock/modify_portfolio.json?symbols={}&pnames={}&category=1".format(
        stock, group_name)
    resp = requests.post(url, headers=headers, cookies=cj)
    if resp.status_code == 200:
        pass


A
if __name__ == '__main__':
    print(get_groups())
    # print(add_groups("zzz"))
    # print(add_stock_to_group("zzz", "SZ300722"))
    print(delete_groups("www"))
