from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlsplit, urlunsplit

import bs4
import gradio as gr
import re
import requests
from bs4 import BeautifulSoup

headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36',
    "Content-Type": "application/json; charset=utf-8"
}


class ImageScrapy(object):

    def __init__(self, full_url, save_root="./"):
        self.full_url = full_url
        self.save_root = Path(save_root)
        parts = urlsplit(full_url)
        self.web_root = urlunsplit((parts.scheme, parts.netloc, '', '', ''))
        self.html = None
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

    def get_html_title(self, soup, img_num: int):
        return re.split(r"[\b\W\b]+", soup.find('title').text)[0] + "[" + str(img_num) + "P]"

    def get_abs_url(self, path):
        return urljoin(self.web_root, path)

    def download_img(self, selector: Callable[[bs4.Tag], bool], img_link_attr: str = "src"):
        html_page = self.get_html()
        soup = BeautifulSoup(html_page, "lxml")
        all_images = soup.find_all(selector)
        if all_images:
            root_path = Path(self.get_html_title(soup, len(all_images)))
            root_path.mkdir(exist_ok=True)
            for idx, img in enumerate(all_images):
                img_url = img.get(img_link_attr)
                if img_url is None:
                    print(f"image tag: {img} can not parse image link.")
                    continue
                path = root_path / f"{idx}{Path(img_url).suffix}"
                print(img_url, path)
                with self.sess.get(img_url, headers=headers, stream=True) as r:
                    with open(path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)

        return len(all_images)


def xyz_selector(tag: bs4.Tag) -> bool:
    return tag.parent.get("id", "") == "conttpc" and tag.name == "img"

def xp1024_selector(tag: bs4.Tag) -> bool:
    return tag.parent.get("id", "") == "read_tpc" and tag.name == "img"


selectors = {
    "xyz": (xyz_selector, "ess-data"),
    "xp1024": (xp1024_selector, "src"),
}


def parse_images(url: str, selector: str) -> int:
    scrapy = ImageScrapy(url)
    return scrapy.download_img(*selectors[selector])


demo = gr.Interface(
    fn=parse_images,
    # inputs="text",
    inputs=[
        gr.Textbox(
            label="Image URL",
            lines=1,
            value="The quick brown fox jumped over the lazy dogs.",
        ),
        gr.Dropdown(
            ["xyz", "xp1024"],
            label="Selector",
            info="Will add more selectors later!"
        ),
    ],
    outputs="text")

if __name__ == '__main__':
    # https://cl.3572x.xyz/htm_mob/2303/7/5601509.html
    # https://yj2303z.xyz/pw/html_data/15/2303/6536994.html
    demo.launch(server_port=9999)
