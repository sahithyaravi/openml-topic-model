from openml import datasets
import pandas as pd
dataset_list = datasets.list_datasets(output_format='dataframe')
ids = []
desc = []
for id in dataset_list['did'][1:5000]:
    try:
        data = datasets.get_dataset(id, download_data=False)
        if data.description:
            ids.append(id)
            data.description.split()
            desc.append()
    except:
        pass
df = pd.DataFrame()
df['id'] = ids
df['text'] = desc
pd.set_option('display.max_colwidth', -1)
print(df['text'].head())
df.to_pickle('df.pkl')
