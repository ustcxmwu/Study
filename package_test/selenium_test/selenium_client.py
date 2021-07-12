from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary


if __name__ == '__main__':
    firefox_binary = FirefoxBinary('C:/Program Files/Mozilla Firefox/firefox.exe')
    driver = webdriver.Firefox(executable_path='C:/geckodriver/geckodriver.exe', firefox_binary=firefox_binary)
    driver.get("http://www.python.org")
    assert "Python" in driver.title
    elem = driver.find_element_by_name("q")
    elem.clear()
    elem.send_keys("pycon")
    elem.send_keys(Keys.RETURN)
    assert "No results found." not in driver.page_source
    driver.close()

