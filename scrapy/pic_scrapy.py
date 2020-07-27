import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin


def get_abs_url(web_root, relative_url):
    return urljoin(web_root, relative_url)


def get_title(soup):
    return soup.find('title').text.strip()


def get_url(url):
    try:
        read = requests.get(url, timeout=2)
        read.raise_for_status()
        read.encoding = read.apparent_encoding
        return read.text
    except Exception as e:
        print(e)


def download_img(html):
    soup = BeautifulSoup(html, "html.parser")
    all_images = soup.find_all('img', attrs={'style': 'cursor: pointer '})
    for idx, img in enumerate(all_images):
        img_url = img['src']
        print(img_url)
        img_url = img_url.replace('thumb_', '')
        img_url = get_abs_url('https://forum.xitek.com', img_url)
        print(img_url)
        root_path = get_title(soup)
        path = os.path.join(root_path, "{}.jpg".format(idx))
        print(path)
        try:
            if not os.path.exists(root_path):
                os.mkdir(root_path)
            if not os.path.exists(path):
                read = requests.get(img_url)
                with open(path, mode='wb') as f:
                    f.write(read.content)
        except:
            print("error")


if __name__ == '__main__':
    html = get_url("https://forum.xitek.com/thread-1890437-1-1-1.html")
    download_img(html)
