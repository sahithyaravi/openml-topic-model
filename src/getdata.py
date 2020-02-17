from openml import datasets
import pandas as pd


class Dataset:
    def __init__(self):
        self.df = None
        self.df_unique = None

    def _remove_duplicates(self):
        """

        :param df: The original list of descriptions
        :return: remove duplicate datasets by choosing the one with longest description.

        """
        grouped = self.df.groupby('name')
        self.df["len"] = self.df["text"].str.len()
        df_new = pd.DataFrame()
        self.df["len"].fillna(0, inplace=True)

        for name, group in grouped:
            idx = group["len"].idxmax()  # index of maximum len
            df_new = pd.concat([df_new, group.loc[[idx]]], ignore_index=True)
        df_new = df_new.loc[df_new["len"] > 50]
        return df_new

    def get_openml_data(self, cache: bool = True):
        """
        get all openml dataset descriptions
        :param cache: If true, the descriptions are fetched from the data folder (already downloaded)
                      If false, the descriptions are fetched from openml for all datasets (will take time)
        :return:
        """
        if cache:
            self.df = pd.read_pickle('../data/df.pkl')
            self.df_unique = self._remove_duplicates()
        else:
            dataset_list = datasets.list_datasets(output_format='dataframe', status='all')
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
                except FileNotFoundError:
                    pass
            self.df = pd.DataFrame()
            self.df['id'] = ids
            self.df['name'] = names
            self.df['text'] = desc
            self.df.to_pickle('../data/df.pkl')
            self.df_unique = self._remove_duplicates()
            self.df_unique.to_pickle('../data/df_unique.pkl')
        return self.df_unique


# pd.set_option('display.expand_frame_repr', False)
# d = Dataset()
# df = d.get_openml_data(cache=True)
# df.to_csv('unique.csv')
# print(df.shape)
# print(df.head(10))
