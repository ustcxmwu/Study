import pandas as pd
from pandas_profiling import ProfileReport


def hushi_num(a):
    if a["15、员数量—护士(1~5)"] == 1:
        return "1~5"
    elif a["15、护士(6~11)"] == 1:
        return "6~11"
    elif a["15、护士(11~20)"] == 1:
        return "11~20"
    elif a["15、护士(21~30)"] == 1:
        return "21~30"
    elif a["15、护士(31~50)"] == 1:
        return "31~50"
    elif a["15、护士(50以上)"] == 1:
        return "50以上"


if __name__ == '__main__':
    df = pd.read_excel('data.xlsx', engine="openpyxl")
    col_name = df.columns.tolist()
    new_col_name = [name.encode('utf-8').decode('utf-8') for name in col_name]
    df.columns = new_col_name
    print(df.columns)
    profile = ProfileReport(df, title="Pandas Profiling Report", pool_size=1)
    profile.to_file("report.html")
    # df["护士人数"] = df.apply(lambda a: hushi_num(a), axis=1)
    # col_name.insert(col_name.index('15、员数量—护士(1~5)'), '护士人数')
    # print(col_name)
    #
    # # df = df[col_name]
    # df = df.reindex(columns=col_name)
    # print(df.head())
    # df.to_excel("df.xlsx")