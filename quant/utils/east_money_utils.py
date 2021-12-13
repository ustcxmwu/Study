import json
import re
from pprint import pprint

import browser_cookie3
import requests
import yaml
from easydict import EasyDict as edict
from requests import Response

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
    with open("config.yaml", mode='r') as f:
        config = yaml.safe_load(f)
    config = edict(config)
    headers = {'content-type': 'application/json',
               'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
               'Referer': 'http://quote.eastmoney.com/'}
    cookies_jar = browser_cookie3.chrome()
    res = requests.get(config.urls.index, headers=headers)
    if res.status_code == 200:
        result = res.text
        for line in result.split("\n"):
            o = re.search(r"var appkey = '([a-z0-9]+)';", line)
            if o:
                config.appkey = o.group(1)
                # print(config.appkey)
    return config, headers, cookies_jar


def parse_cookies(cookie: str):
    # cookies = dict()
    # for line in cookie.split("; "):
    #     k, v = line.split("=")
    #     cookies[k] = v
    # return cookies
    cj = browser_cookie3.chrome()  # firefox可以替换为browser_cookie3.firefox()
    return cj


def get_groups():
    config, headers, cj = load_config()
    url = config.urls.webouter + config.urls.groups
    url = url.format(config.appkey)
    resp = requests.get(url, headers=headers, cookies=cj)

    _, value = parse_resp(resp, key='ginfolist')
    return value


def get_group_id(gname: str):
    groups = get_groups()
    for ginfo in groups:
        if gname == ginfo['gname']:
            return ginfo['gid']
    return None


def add_group(gname: str):
    if get_group_id(gname) is not None:
        raise ValueError("自选组: {}已存在,请更换名字后继续.".format(gname))
    config, headers, cookies = load_config()
    url = config.urls.webouter + config.urls.add_group
    url = url.format(config.appkey, gname)
    resp = requests.get(url, headers=headers, cookies=cookies)
    state, result = parse_resp(resp)
    if state != 0:
        raise ValueError(result)
    return edict(result)


def rename_group(old_name: str, new_name: str):
    gid = get_group_id(old_name)
    if gid is None:
        raise ValueError("自选组: {} 不存在".format(old_name))
    config, headers, cookies = load_config()
    url = config.urls.webouter + config.urls.rename_group
    url = url.format(config.appkey, gid, new_name)
    resp = requests.get(url, headers=headers, cookies=cookies)
    state, result = parse_resp(resp)
    if state != 0:
        raise ValueError(result)
    return edict(result)


def del_group(gname: str):
    gid = get_group_id(gname)
    if gid is None:
        raise ValueError("自选组: {} 不存在, 无需删除".format(gname))
    config, headers, cookies = load_config()
    url = config.urls.webouter + config.urls.del_group
    url = url.format(config.appkey, gid)
    resp = requests.get(url, headers=headers, cookies=cookies)
    state, result = parse_resp(resp)
    if state != 0:
        raise ValueError(result)
    return edict(result)


def get_stocks(gname: str):
    gid = get_group_id(gname)
    if gid is None:
        raise ValueError("自选组: {} 不存在".format(gname))
    config, headers, cookies = load_config()
    url = config.urls.webouter + config.urls.stocks
    url = url.format(config.appkey, gid)
    resp = requests.get(url, headers=headers, cookies=cookies)
    state, result = parse_resp(resp)
    if state != 0:
        raise ValueError(result)
    return edict(result)


def add_stock(code, gname: str):
    gid = get_group_id(gname)
    if gid is None:
        raise ValueError("自选组: {} 不存在".format(gname))
    config, headers, cookies = load_config()
    url = config.urls.webouter + config.urls.add_to_group
    codes = to_eastmoney_code(code)
    url = url.format(config.appkey, gid, codes)
    resp = requests.get(url, headers=headers, cookies=cookies)
    state, result = parse_resp(resp)
    if state != 0:
        raise ValueError(result)
    return edict(result)


def del_from_group(code: str, gname: str):
    gid = get_group_id(gname)
    if gid is None:
        raise ValueError("自选组: {} 不存在".format(gname))
    config, headers, cookies = load_config()
    url = config.urls.webouter + config.urls.del_from_group
    codes = to_eastmoney_code(code)
    url = url.format(config.appkey, gid, codes)
    resp = requests.get(url, headers=headers, cookies=cookies)
    state, result = parse_resp(resp)
    if state != 0:
        raise ValueError(result)
    return edict(result)


def mod_stock_group(code: str, old_group_name: str, new_group_name: str):
    gid = get_group_id(old_group_name)
    gid2 = get_group_id(new_group_name)
    if gid is None:
        raise ValueError("自选组: {} 不存在".format(old_group_name))
    if gid2 is None:
        raise ValueError("自选组: {} 不存在".format(new_group_name))
    config, headers, cookies = load_config()
    url = config.urls.webouter + config.urls.mod_group
    codes = to_eastmoney_code(code)
    url = url.format(config.appkey, gid, gid2, codes)
    resp = requests.get(url, headers=headers, cookies=cookies)
    state, result = parse_resp(resp)
    if state != 0:
        raise ValueError(result)
    return edict(result)


def add_qingze_group(gname: str, filename: str):
    gid = get_group_id(gname)
    if gid is not None:
        del_group(gname)
    add_group(gname)
    gid = get_group_id(gname)
    config, headers, cookies = load_config()
    urlf = config.urls.webouter + config.urls.add_to_group
    qingze_codes = list()
    with open(filename, mode='r') as f:
        for line in f.readlines():
            qingze_codes.append(line.strip("\n"))
    for code in qingze_codes:
        codes = to_eastmoney_code(code.strip("\n"))
        url = urlf.format(config.appkey, gid, codes)
        resp = requests.get(url, headers=headers, cookies=cookies)
        state, result = parse_resp(resp)
        if state != 0:
            print("add stock: {} error: {}".format(code, result))
        else:
            print(result)


def main():
    pprint(get_groups())
    # print(del_group("清则洞察"))
    # print(add_qingze_group("清则洞察9月", "清则洞察9月.txt"))
    # print(add_qingze_group("清则ETF", "ETF导入.txt"))
    # print(add_qingze_group("清则好公司1", "好公司导入.txt"))
    # print(get_stocks("清则好公司"))
    # print(get_groups())
    # print(get_stocks("xx"))
    # print(get_group_id("清则群"))
    # print(get_group_id("xxxxx"))
    # print(rename_group("xxxxx", "yyyyyy"))
    # print(get_group_id("yyyyyy"))


if __name__ == '__main__':
    main()
