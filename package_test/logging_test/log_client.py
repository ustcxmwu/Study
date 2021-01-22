#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.
import yaml
import logging
import logging.config
import logging.handlers


def init_logging():
    with open("./log.yml", mode='r') as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config=config)


def get_named_logger(name: str):
    logger = logging.getLogger("console.{}".format(name))
    fh = logging.handlers.RotatingFileHandler('./log/{}.log'.format(name))
    log_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(lineno)04d | %(message)s')
    fh.setFormatter(log_formatter)
    logger.addHandler(fh)
    return logger




class Report(object):

    def __init__(self, idx: int):
        self.idx = idx
        self.logger = get_named_logger("re{}".format(self.idx))


if __name__ == '__main__':
    init_logging()
    r1 = Report(1)
    r1.logger.info("XXXXXXXXXXXXXXXXXXXXX")
    r2 = Report(2)
    r2.logger.info("YYYYYYYYYYYYYYYY")

    d = {
        "a": (1, 2),
        "b": (3, 4)
    }
    for k, (x1, x2) in d.items():
        print(x1, x2)