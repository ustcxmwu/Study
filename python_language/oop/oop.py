import abc  # 利用abc模块实现抽象类
from typing import Type


class AllFile(metaclass=abc.ABCMeta):
    all_type = 'file'

    @abc.abstractmethod  # 定义抽象方法，无需实现功能
    def read(self):
        '子类必须定义读功能'
        pass

    @abc.abstractmethod  # 定义抽象方法，无需实现功能
    def write(self):
        '子类必须定义写功能'
        pass


class Txt(AllFile):
    def read(self):
        print('文本数据的读取方法')

    def write(self):
        print('文本数据的读取方法')


class Sata(AllFile):
    def read(self):
        print('硬盘数据的读取方法')

    def write(self):
        print('硬盘数据的读取方法')


class Process(AllFile):
    def read(self):
        print('进程数据的读取方法')

    def write(self):
        print('进程数据的读取方法')


def dis(a: Type[AllFile]):
    a.read()
    a.write()


if __name__ == '__main__':
    wenbenwenjian = Txt()
    dis(wenbenwenjian)
    # yingpanwenjian=Sata()
    # jinchengwenjian=Process()
    #
    # #这样大家都是被归一化了,也就是一切皆文件的思想
    # wenbenwenjian.read()
    # yingpanwenjian.write()
    # jinchengwenjian.read()
    #
    # print(wenbenwenjian.all_type)
    # print(yingpanwenjian.all_type)
    # print(jinchengwenjian.all_type)
    # # print(AllFile().all_type)
    #
    # print(np.arange(0, 2*math.pi, math.pi/6))
    # print(np.linalg.norm(np.array([1, 1]) - np.array([2, 2])))
    # a = np.zeros(28)
    # b = np.array([3, 4])
    # a[0:2] = b[:]
    # print(a)
    # print(np.pi/6*np.arange(-3, 4))
