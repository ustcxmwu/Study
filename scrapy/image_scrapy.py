import os
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup


class ImageScrapy(object):

    def __init__(self, full_url, save_root="./", filters=None):
        self.full_url = full_url
        self.save_root = save_root
        parts = urlsplit(full_url)
        self.web_root = urlunsplit((parts.scheme, parts.netloc, '', '', ''))
        self.html = None
        self.filters = filters

    def get_html(self):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
            read = requests.get(self.full_url, timeout=2, headers=headers)
            read.raise_for_status()
            read.encoding = read.apparent_encoding
            self.html = read.text
        except Exception as e:
            print(e)

    def get_html_title(self, soup):
        # return soup.find('title').text.strip()
        return soup.find('title').text.split(' ')[0].strip()

    def get_abs_url(self, path):
        return urljoin(self.web_root, path)

    def download_img(self):
        self.get_html()
        soup = BeautifulSoup(self.html, "html.parser")
        # all_images = soup.find_all('img', attrs={'style': 'cursor: pointer '})
        all_images = soup.find_all('img')
        for filter_func in self.filters:
            all_images = filter(filter_func, all_images)
        if all_images:
            root_path = self.get_html_title(soup)
            try:
                if not os.path.exists(root_path):
                    os.mkdir(root_path)
            except:
                print("can not create root directory")
        for idx, img in enumerate(all_images):
            # img_url = img['src']
            img_url = img['ess-data']
            # print(img_url)
            # img_url = img_url.replace('thumb_', '')
            img_url = self.get_abs_url(img_url)
            print(img_url)
            img_ext = img_url.split('.')[-1]
            path = os.path.join(root_path, "{}.{}".format(idx, img_ext))
            print(path)
            try:
                if not os.path.exists(path):
                    read = requests.get(img_url)
                    with open(path, mode='wb') as f:
                        f.write(read.content)
            except:
                print("cannot save image")


if __name__ == '__main__':
    # url = "https://forum.xitek.com/thread-1890437-1-1-1.html"
    url = "https://forum.xitek.com/thread-1926536-1-1.html"
    # url = "https://cl.ht52.xyz/htm_data/2007/7/4022171.html"
    # scrapy = ImageScrapy(url, filters=[lambda x: x.get("ess-data")])
    scrapy = ImageScrapy(url, filters=[lambda x: x])
    scrapy.download_img()
