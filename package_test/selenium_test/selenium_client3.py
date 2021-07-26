from selenium import webdriver
from lxml import etree
from time import sleep

bro = webdriver.Chrome(executable_path='D:\Program Files (x86)\Chrome\chromedriver.exe')
bro.get('http://scxk.nmpa.gov.cn:81/xk/')
# bro.switch_to_alert().accept()


def get_page(page_text=None):
    if page_text is None:
        page_text = bro.page_source
        # h1 = bro.current_window_handle
        # print(h1)
    # 获取新数据示例
    tree = etree.HTML(page_text)
    li_list = tree.xpath('//ul[@id="gzlist"]/li')
    for li in li_list:
        name = li.xpath('./dl/@title')[0]
        print(name)

    bro.find_element_by_xpath('//*[@id="pageIto_next"]').click()
    sleep(5)
    bro.switch_to.window(bro.window_handles[0])
    # h2 = bro.current_window_handle
    # print(h2)
    page_text_ = bro.page_source
    if page_text_ == page_text:
        quit()
    else:
        # 递归
        get_page(page_text_)


if __name__ == '__main__':
    get_page()
    sleep(5)
    bro.quit()
