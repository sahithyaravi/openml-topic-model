from openml import datasets
import pandas as pd


def remove_duplicates(df):
    """

    :param df: The original list of descriptions
    :return: remove duplicate datasets by choosing the one with longest description.

    """
    grouped = df.groupby('name')
    df["len"] = df["text"].str.len()
    df_new = pd.DataFrame()
    df["len"].fillna(0, inplace=True)

    for name, group in grouped:
        idx = group["len"].idxmax()  # index of maximum len
        print ("idx", idx)
        df_new = pd.concat([df_new, group.loc[[idx]]], ignore_index=True)

    df_new.to_pickle('../data/df_unique.pkl')
    return df_new


def get_openml_data(cache: bool = True):
    """
    get all openml dataset descriptions
    :param cache: If true, the descriptions are fetched from the data folder (already downloaded)
                  If false, the descriptions are fetched from openml for all datasets (will take time)
    :return:
    """
    if cache:
        pd.read_pickle('../data/df_unique.pkl')
    else:
        dataset_list = datasets.list_datasets(output_format='dataframe',status='all')
        print(dataset_list.shape)
        ids = []
        names = []
        desc = []
        for did in dataset_list['did']:
            try:
                data = datasets.get_dataset(did, download_data=False)
                if data.description is not None and data.name is not None:
                    ids.append(did)
                    names.append(data.name)
                    desc.append(data.description + " " + data.name + " ")
            except:
                pass
        df = pd.DataFrame()
        df['id'] = ids
        df['name'] = names
        df['text'] = desc
        df.to_pickle('../data/df.pkl')
        df = remove_duplicates(df)
    return df


# # Test this file
# df = get_openml_data(cache=False)
# print(df.shape)
