from helium import *
from selenium import webdriver


if __name__ == '__main__':
    driver = webdriver.Chrome(executable_path='D:\Program Files (x86)\Chrome\chromedriver.exe')
    set_driver(driver)

    # start_chrome("http://quote.eastmoney.com/zixuan/")