#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import pathlib
import os
import time
from pathlib import Path
import pandas as pd


if __name__ == '__main__':
    # a = pathlib.Path("..")
    # a_files = []
    # for file in a.rglob("^[^.]*"):
    #     if file.is_file():
    #         a_files.append(str(file.relative_to("..")))
    # b_files = []
    # for root, dirs, files in os.walk(".."):
    #     for file in files:
    #         b_files.append(os.path.join(root, file).replace("..\\", ""))
    # print(len(a_files))
    # print(a_files)
    # print("\n\n\n")
    # print(len(b_files))
    # print(b_files)
    #
    # print(a_files == b_files)

    # for i in range(100):
    #     print("\r{}".format(i).center(80, "="), end="")
    #     time.sleep(0.2)

    # files = [Path("a.data-00000-of-00001"), Path('b.index'), Path("c.meta"), Path("d"), Path("f.yml")]
    # print(sorted(files, key= lambda x:  x.suffix))

    # a = [['刘玄德', '男', '语文', 98.], ['刘玄德', '男', '体育', 60.], ['关云长', '男', '数学', 60.], ['关云长', '男', '语文', 100.]]
    # af = pd.DataFrame(a, columns=["姓名", "性别", "科目", "成绩"])
    # print(af)
    # af.insert(0, "group", "a")
    # af.reset_index()
    # print(af)
    # # af["index"] = af.index
    # af.insert(0, "index", af.index)
    # af = af.set_index(["group", "index"])
    # print(af)
    # writer = pd.ExcelWriter("a.xlsx")
    # af.to_excel(writer)
    # writer.save()

    # af.set_index()
    # af.reset_index()
    # b = af.T
    # print(b)
    # a = Path("a/b/c")
    # print(a)
    # b = a.joinpath("c/d")
    # print(b)

    files = {
        "f.yml": 5,
        "a.lock": 0,
        "c.meta": 3,
        "a.data-00000-of-00001": 1,
        'b.index': 2,
        "d": 4,
    }
    sync_files = [(name, md5) for name, md5 in files.items() if not name.endswith(".lock")]
    print(sync_files)
    # s_files = sorted(sync_files, key=lambda f: Path(f[0]).suffix)
    # print(s_files)
    sync_files.sort(key=lambda f: Path(f[0]).suffix)
    print(sync_files)

