#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import pdfkit


if __name__ == "__main__":
    url = "https://communication.portal.netease.com/honor/application/246"
    pdfkit.from_url(url, 'a.pdf')