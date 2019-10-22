from openml import datasets
import pandas as pd
dataset_list = datasets.list_datasets(output_format='dataframe')
ids = []
desc = []
for id in dataset_list['did'][1:500]:
    data = datasets.get_dataset(id, download_data=False)
    if data.description:
        ids.append(id)
        desc.append(data.description)
df = pd.DataFrame()
df['id'] = ids
df['text'] = desc
pd.set_option('display.max_colwidth', -1)
print(df['text'].head())
df.to_pickle('df.pkl')
