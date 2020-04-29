import pandas as pd
import numpy as np

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt


if __name__ == '__main__':


    url = "https://www.google.com/flights/explore/#explore;f=JFK,EWR,LGA;t=HND,NRT,TPE,HKG," \
                "KIX;s=1;li=8;lx=12;d=2016-04-01"

    driver = webdriver.PhantomJS()

    dcap = dict(DesiredCapabilities.PHANTOMJS)
    dcap["phantomjs.page.settings.userAgent"] = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 ("
                                                 "KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36")

    driver = webdriver.PhantomJS(desired_capabilities=dcap, service_args=['--ignore-ssl-error=true'])

    driver.implicitly_wait(20)
    driver.get(url)
    driver.save_screenshot(r'flight_explorer.png')


