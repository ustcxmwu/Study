from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver


if __name__ == '__main__':
    browser = webdriver.Chrome(ChromeDriverManager().install())
    browser.get("https://www.baidu.com")
    print(browser.title)