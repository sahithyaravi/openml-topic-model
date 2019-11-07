import pandas as pd

df = pd.read_pickle("df.pkl")
grouped = df.groupby('name')
df["len"] =  df["name"].str.len()
df_new = pd.DataFrame()
df["len"].fillna(0, inplace=True)
for name,group in grouped:
    idx = group["len"].idxmax()
    print(idx)
    df_new = pd.concat([df_new, group.ix[[idx]]], ignore_index=True)
    print(df_new)


print(df_new.shape, df.shape)
df_new.to_pickle('df_unique.pkl')
df_new.to_csv('df_unique.csv')