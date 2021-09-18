from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

import win32gui
import win32con
from win32com.client import Dispatch
from win32gui import GetClassName

import time
import logging
import argparse

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutoChecker(object):
    ShellWindowsCLSID = '{9BA05972-F6A8-11CF-A442-00A0C90A8F39}'

    @staticmethod
    def wait_for_player_ready(timeout=0):
        cnt = 0
        while cnt <= timeout:
            sws = Dispatch(AutoChecker.ShellWindowsCLSID)
            cnt += 1
            for sw in sws:
                if GetClassName(sw.HWND) == 'IEFrame':
                    logger.debug(f"IE窗口名：{sw.LocationName}， url：{sw.LocationURL}")

                    url = str(sw.LocationURL)
                    if url.find("coursetitle") > 0 and url.find("courseid=") > 0:
                        return True

            if cnt <= timeout:
                time.sleep(1)

        return False

    @staticmethod
    def wait_for_player_close(timeout=0):
        cnt = 0
        while cnt <= timeout:
            sws = Dispatch(AutoChecker.ShellWindowsCLSID)
            cnt += 1
            for sw in sws:
                if GetClassName(sw.HWND) == 'IEFrame':
                    logger.debug(f"IE窗口名：{sw.LocationName}， url：{sw.LocationURL}")

                    url = str(sw.LocationURL)
                    if url.find("coursetitle") > 0 and url.find("courseid=") > 0:
                        return False

            if cnt <= timeout:
                time.sleep(1)

        return True

    @staticmethod
    def close_player_windows():
        sws = Dispatch(AutoChecker.ShellWindowsCLSID)
        for sw in sws:
            if GetClassName(sw.HWND) == 'IEFrame':
                logger.debug(f"IE窗口名：{sw.LocationName}， url：{sw.LocationURL}")

                url = str(sw.LocationURL)
                if url.find("coursetitle") > 0 and url.find("courseid=") > 0:
                    win32gui.PostMessage(sw.HWND, win32con.WM_CLOSE, 0, 0)
                    return True

        return False

    @staticmethod
    def close_popup_windows():
        target_hwnd = []

        def foo(hwnd, nouse):
            if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
                if win32gui.GetWindowText(hwnd).find("来自网页的消息") >= 0:
                    target_hwnd.append(hwnd)

        win32gui.EnumWindows(foo, 0)
        if len(target_hwnd) == 1:
            win32gui.PostMessage(target_hwnd[0], win32con.WM_CLOSE, 0, 0)
            logger.debug("发现了目标窗口，已关闭！")
            return True
        else:
            logger.debug("未发现任何目标窗口 或 出现干扰！")
            return False


class AutoPlayer(object):
    def __init__(self, account, pwd):
        capabilities = DesiredCapabilities.INTERNETEXPLORER

        # delete platform and version keys
        capabilities.pop("platform", None)
        capabilities.pop("version", None)

        # start an instance of IE
        self.driver = webdriver.Ie(executable_path="C:\\Program Files (x86)\\Internet Explorer\\IEDriverServer.exe",
                              capabilities=capabilities)

        self.driver.implicitly_wait(20)
        self.driver.get("https://pro.learning.gov.cn/")
        self.account = account
        self.pwd = pwd

    def login(self):
        self.driver.implicitly_wait(20)
        self.driver.get("https://pro.learning.gov.cn/")
        time.sleep(1)
        self.driver.get("https://puser.zjzwfw.gov.cn/sso/usp.do?action=ssoLogin&servicecode=xxxgx")
        time.sleep(2)

        # js_code = """document.getElementsByName("loginname")[0].value = "loginname"\n
        # document.getElementsByName("loginpwd").[0]value = "password"\n
        # document.getElementById("submit").click()"""
        # self.driver.execute_script(js_code)

        self.driver.find_element_by_name("loginname").send_keys(self.account)
        time.sleep(1)
        self.driver.find_element_by_name("loginpwd").send_keys(self.pwd)
        time.sleep(1)

        self.driver.find_element_by_id("submit").click()
        time.sleep(2)

        self.driver.find_element_by_class_name("login-btn2").click()
        time.sleep(2)

    def _get_chosen_lessons_page_num(self):
        self.driver.get("https://pro.learning.gov.cn/my/")
        time.sleep(2)

        page_number = self.driver.find_element_by_class_name("page-turning").find_element_by_class_name("text").text

        return int(page_number[1:-1])

    def _get_chosen_lessons_one_page(self, page_index):

        url = f"https://pro.learning.gov.cn/my/?offset={page_index}&"
        self.driver.get(url)
        time.sleep(1)

        # 定位到table，并获得table中所有得tr元素
        lesson_table = self.driver.find_element_by_class_name("course-table")
        rows = lesson_table.find_elements_by_tag_name('tr')

        lessons = []
        for row in rows:
            try:
                columns = row.find_elements_by_tag_name('td')
            except Exception as e:
                print(e)
                continue

            if columns is None or len(columns) == 0:
                continue

            url_str = columns[0].find_element_by_tag_name("a").get_attribute("href")
            title_str = columns[0].find_element_by_tag_name("a").get_attribute("title")
            if url_str:
                lessons.append((url_str, title_str))

        return lessons

    def get_chosen_lessons(self):
        page_num = self._get_chosen_lessons_page_num()

        logger.info(f"一共选了{page_num}页课程")

        lessons = []
        for i in range(1, page_num+1,1):
            _ls = self._get_chosen_lessons_one_page(i)
            lessons.extend(_ls)

        return lessons

    def begin_lesson(self, lesson):
        url = lesson[0]
        title = lesson[1]

        self.driver.get(url)
        time.sleep(3)
        self.driver.find_element_by_class_name("btn").click()

        _ready = False
        while not _ready:
            _ready = AutoChecker.wait_for_player_ready(timeout=10)
            logger.info(f"等待页面开始播放")

        logger.info(f"开始播放:{title}")
        begin_time = time.time()

        popup_closed_time = 0
        while True:
            closed = AutoChecker.wait_for_player_close()
            if not closed:
                popup_closed = AutoChecker.close_popup_windows()
                if popup_closed:
                    popup_closed_time += 1
                    logger.info(f"关闭了第{popup_closed_time}个弹窗")

                # 如果单个视频超过一小时，还没有播放成功， 有可能卡住或者出现问题， 关闭网页窗口
                if (time.time() - begin_time) > 3600:
                    if AutoChecker.close_player_windows():
                        logger.info("播放时间超过1小时， 关闭播放页面")

                time.sleep(15)
            else:
                logger.info(f"播放完成:{title}")
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--account', help='account info', type=str, required=True)
    parser.add_argument('-p', '--password', help='password info', type=str, required=True)
    args = parser.parse_args()

    auto_player = AutoPlayer(account=args.account, pwd=args.password)
    auto_player.login()

    lessons = auto_player.get_chosen_lessons()
    logger.info(f"一共获取{len(lessons)}门课程， 开始顺序播放！")

    for lesson in lessons:
        logger.info(f"准备播放{lesson[0]}")
        auto_player.begin_lesson(lesson)















