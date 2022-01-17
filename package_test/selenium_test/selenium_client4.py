from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


if __name__ == '__main__':
    browser = webdriver.Chrome()
    browser.get('http://www.google.com')

    print(browser.title)
    elem = browser.find_element(By.NAME, 'q')  # Find the search box
    elem.send_keys('seleniumhq' + Keys.RETURN)

    browser.quit()