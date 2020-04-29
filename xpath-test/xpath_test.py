from lxml import etree
import numpy as np
import random
import logging
from recordtype import recordtype
import math
import pyglet
import sys


wb_data = """
            <div>
                <ul>
                     <li class="item-0"><a href="link1.html">first item</a></li>
                     <li class="item-1"><a href="link2.html">second item</a></li>
                     <li class="item-inactive"><a href="link3.html">third item</a></li>
                     <li class="item-1"><a href="link4.html">fourth item</a></li>
                     <li class="item-0"><a href="link5.html">fifth item</a>
                 </ul>
             </div>
            """

def xpath_unit1():
    html = etree.HTML(wb_data)
    print(html)
    print('================================================')
    print('================================================')
    print('================================================')
    result = etree.tostring(html)
    print(result.decode("utf-8"))


def xpath_unit2():
    html = etree.HTML(wb_data)
    html_data = html.xpath('/html/body/div/ul/li/a')
    for i in html_data:
        print(i.text)


def xpath_unit3():
    html = etree.HTML(wb_data)
    html_data = html.xpath('/html/body/div/ul/li/a/text()')
    print(html)
    for i in html_data:
        print(i)


class Com:

    def __init__(self):
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename='parser_result.log',
                            filemode='w')
        self.logger = logging.getLogger('Eclipse_Isle_2D')

    def log(self):
        self.logger.info('wwwwww')
        self.logger.debug('debug')
        self.logger.warning('warning')
        self.logger.error('error')


def func1():
    legal = dict(zip(range(5), [True] * 5))
    print(legal)

if __name__ == '__main__':
    # pyglet.window.Window(800, 600, display=None)
    print((1,1)==(1,1))
    print((1,1)==(1,2))




