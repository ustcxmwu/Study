import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains


def main():
    browser = webdriver.Chrome(ChromeDriverManager().install())
    browser.get("https://www.google.com")
    print(browser.title)

    elem = browser.find_element(By.NAME, 'q')  # Find the search box
    elem.send_keys('seleniumhq' + Keys.RETURN)

    browser.quit()


if __name__ == '__main__':
    browser = webdriver.Chrome(ChromeDriverManager().install())
    browser.get("http://quote.eastmoney.com/zixuan/")
    print(browser.title)

    elem = browser.find_element(By.LINK_TEXT, '登录')
    elem.click()
    time.sleep(1)

    browser.switch_to.frame("frame_login")
    browser.execute_script("document.getElementsByClassName('login_content')[0].style.display = 'block';")
    browser.execute_script("document.getElementsByClassName('account title')[0].setAttribute('class', 'account title current');")
    browser.execute_script("document.getElementsByClassName('msg title current')[0].setAttribute('class', 'msg title');")
    
    # action = ActionChains(browser)
    # userpd =browser.find_element(By.LINK_TEXT, "账号密码登录")
    # action.move_to_element(userpd).perform()
    login = browser.find_element(By.NAME, 'login_email')
    login.send_keys("13405863289")
    login.send_keys(Keys.ENTER)
    passwd = browser.find_element(By.NAME, 'login_password')
    passwd.send_keys("kdwxm1309")
    passwd.send_keys(Keys.ENTER)
    userpd =browser.find_element(By.CSS_SELECTOR, '.selectbox.unselected')
    ActionChains(browser).move_to_element(userpd).click().perform()
    # browser.execute_script("document.getElementsByClassName('selectbox unselected')[0].style.display = 'none';"
    #                        "document.getElementsByClassName('selectbox selected')[0].style.display = 'block';")
    # notice = browser.find_element(By.CLASS_NAME, 'checkbox')
    # notice.click()
    submit = browser.find_element(By.ID, 'btn_login')
    submit.submit()
    # browser.execute_script("document.getElementById('login_content')[0].style.display = 'block';")
    # time.sleep(1)
    browser.implicitly_wait(5)
    check = browser.find_element(By.ID, "divCaptcha")
    ActionChains(browser).move_to_element(check).click().perform()
    browser.implicitly_wait(20)

    cookie = browser.get_cookies()
    print(cookie)
    # notice.select_by_value()


    # elem.send_keys('seleniumhq' + Keys.RETURN)
