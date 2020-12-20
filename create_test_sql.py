import pandas as pd
df = pd.read_csv('results/NOUNADJ_filter_0.7/resultdf.csv')
keys = list(range(-1, 10))
values = ['unknown', 'trade', 'medicine', 'drug', 'friedman',
          'drug', 'image', 'analyis', 'genetics',
          'pattern', 'drug/chemical']
map_dict = dict(zip(keys, values))

df_new = pd.DataFrame()
df_new["id"] = df["id"]
df_new["topic"] = df["best_topic"].map(map_dict)
df_new["uploader"] = 1
df_new["date"] = '2017-03-24 19:36:11'
df_new["processed"] = df["processed"]
df_new = df_new[df_new["id"] < 101]
df_new.to_csv("100.csv")

TARGET = 'dataset_topic'
# sql_texts = []
# for index, row in df.iterrows():
#     sql_texts.append(
#         'INSERT INTO ' + TARGET + ' (' + str(', '.join(df.columns)) + ') VALUES ' + str(tuple(row.values)))
# print(sql_texts)