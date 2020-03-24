from openml import datasets
import pandas as pd


class Dataset:
    def __init__(self):
        self.df = None
        self.df_unique = None

    def get_dataset(self):
        """

        :return: unique dataset descriptions with length min=50
        """
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
            except:
                pass
        self.df = pd.DataFrame()
        self.df['id'] = ids
        self.df['name'] = names
        self.df['text'] = desc
        self.df_unique = self._remove_duplicates()
        return self.df_unique

    def _remove_duplicates(self):
        """
        :return: remove duplicates by choosing the one with longest description.

        """
        grouped = self.df.groupby('name')
        self.df["len"] = self.df["text"].str.len()
        df_new = pd.DataFrame()
        self.df["len"].fillna(0, inplace=True)

        for name, group in grouped:
            idx = group["len"].idxmax()  # index of maximum len
            df_new = pd.concat([df_new, group.loc[[idx]]], ignore_index=True)
        return df_new



