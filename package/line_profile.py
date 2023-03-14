import cgi
import time
from line_profiler import LineProfiler


def aaa():
    print('hello!test2()')


def bbb():
    html = '''<script>alert("you are a good boy!&I like you")</scrpit>'''
    aaa()
    escape_html = cgi.escape(html)
    for item in range(5):
        time.sleep(1)
    print(escape_html)


if __name__ == '__main__':
    lp = LineProfiler()
    # 同时显示函数每行所用时间和调用函数每行所用时间，加入add_function()
    # lp.add_function(aaa)
    lp_wrap = lp(bbb)
    # 如果被测函数有入参，下面一行为 lp_wrap(被测函数入参)
    lp_wrap()
    lp.print_stats()
