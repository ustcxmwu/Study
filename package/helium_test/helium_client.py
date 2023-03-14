from helium import *
from selenium import webdriver


if __name__ == '__main__':
    # driver = webdriver.Chrome(executable_path='D:\Program Files (x86)\Chrome\chromedriver.exe')
    # set_driver(driver)

    # start_chrome("http://quote.eastmoney.com/zixuan/")

    start_chrome('github.com/login')
    write('ustcxmwu', into='Username')
    write('ustcwxm1309', into="Password")
    click('Sign in')