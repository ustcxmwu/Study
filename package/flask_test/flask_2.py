#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return "hello world"


if __name__ == '__main__':
    app.run(port=6001)