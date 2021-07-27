import browser_cookie3
import requests


def main():
    cj = browser_cookie3.chrome()  # firefox可以替换为browser_cookie3.firefox()
    r = requests.get("http://quote.eastmoney.com/zixuan/", cookies=cj)
    key = r.get('appkey')
    print(r)


if __name__ == "__main__":
    main()