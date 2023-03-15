from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import gradio as gr
import requests
from bs4 import BeautifulSoup

headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36',
    "Content-Type": "application/json; charset=utf-8"
}

class ImageScrapy(object):

    def __init__(self, full_url, save_root="./", filters=None):
        self.full_url = full_url
        self.save_root = Path(save_root)
        parts = urlsplit(full_url)
        self.web_root = urlunsplit((parts.scheme, parts.netloc, '', '', ''))
        self.html = None
        self.filters = filters
        self.sess = requests.session()
        self.sess.keep_alive = False

    def get_html(self):
        try:
            read = self.sess.get(self.full_url, timeout=5, headers=headers)
            read.raise_for_status()
            read.encoding = read.apparent_encoding
            return read.text
        except Exception as e:
            print(e)

    def get_html_title(self, soup):
        # return soup.find('title').text.strip()
        return soup.find('title').text.split(' ')[0].strip()

    def get_abs_url(self, path):
        return urljoin(self.web_root, path)

    def download_img(self):
        html_page = self.get_html()
        soup = BeautifulSoup(html_page, "lxml")
        # soup = BeautifulSoup(html_page, "html5lib")
        # soup = BeautifulSoup(self.html, "html.parser")
        # all_images = soup.find_all('img', attrs={'style': 'cursor: pointer '})
        all_images = soup.find(id='conttpc').find_all(name="img")
        if self.filters is not None:
            for filter_func in self.filters:
                all_images = filter(filter_func, all_images)
        if all_images:
            root_path = Path(self.get_html_title(soup))
            root_path.mkdir(exist_ok=True)
            for idx, img in enumerate(all_images):
                # img_url = img['src']
                img_url = img['ess-data']
                # print(img_url)
                img_url = img_url.replace('thumb_', '')
                img_url = self.get_abs_url(img_url)
                print(img_url)
                img_ext = img_url.split('.')[-1]
                path = root_path / "{}.{}".format(idx, img_ext)
                print(path)
                with self.sess.get(img_url, headers=headers, stream=True) as r:
                    with open(path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)

        return len(all_images)


def parse_image(url: str):
    scrapy = ImageScrapy(url)
    return scrapy.download_img()


if __name__ == '__main__':
    # url = "https://forum.xitek.com/thread-1959111-1-1.html"
    demo = gr.Interface(fn=parse_image, inputs="text", outputs="text")
    demo.launch()
