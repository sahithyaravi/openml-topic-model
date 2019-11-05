from openml import datasets
import pandas as pd
dataset_list = datasets.list_datasets(output_format='dataframe')
ids = []
names = []
desc = []
for id in dataset_list['did']:
    try:
        data = datasets.get_dataset(id, download_data=False)
        features = [vars(data.features[i])['name'] for i in range(0, len(data.features))]
        if len(data.description) > 50:
            ids.append(id)
            names.append(data.name)
            desc.append(data.description + " " + data.name+ " " )
        else:
            print(id)
            desc.append(data.description + data.name + " ".join(features))
            ids.append(id)
            names.append(data.name)


    except:
        pass
df = pd.DataFrame()
df['id'] = ids
df['name'] = names
df['text'] = desc
pd.set_option('display.max_colwidth', -1)
df.to_pickle('df.pkl')
