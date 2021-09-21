import json

import browser_cookie3
import requests
from requests import Response


# def get_symbol(code: str):
#     code = code.strip()
#     pre = code[:3]
#     if pre in ["300", "000", "200"]:
#         return "SZ" + code
#     elif pre in ["601", "602", "603", "605", "900", "688", "002"]:
#         return "SH" + code


def get_symbol(code: str):
    code = code.strip()
    pre = int(code)
    if pre < 333333:
        return "SZ" + code
    else:
        return "SH" + code


def load_config():
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36",
        "Referer": "https://xueqiu.com/"
    }
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


def add_stocks(group_name: str, filename: str):
    qingze_codes = []
    with open(filename, mode='r') as f:
        for line in f.readlines():
            qingze_codes.append(get_symbol(line))
    for symbol in qingze_codes:
        add_stock_to_group(group_name, symbol)


if __name__ == '__main__':
    print(get_groups())
    # print(add_groups("zzz"))
    # print(add_stock_to_group("zzz", "SZ300722"))
    # print(delete_groups("www"))
    # add_stocks("清则9月", "清则洞察9月.txt")
    add_stocks("清则ETF", "ETF导入.txt")
