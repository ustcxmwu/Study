#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

from keras.applications.vgg16 import VGG16

if __name__ == '__main__':
    model = VGG16()
    print(model.summary())

