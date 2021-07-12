from selenium import webdriver
import time


def main():
    browser = webdriver.Chrome(executable_path="D:\Program Files (x86)\Chrome\chromedriver.exe")

    website_URL = "https://www.google.co.in/"
    browser.get(website_URL)

    refreshrate = int(3)

    # This would keep running until you stop the compiler.
    while True:
        time.sleep(refreshrate)
        browser.refresh()


if __name__ == "__main__":
    main()
