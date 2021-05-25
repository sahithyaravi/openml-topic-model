import openml
from openml import datasets
import pandas as pd
from collections import defaultdict


class Dataset:
    def __init__(self, config='production'):
        self.df = None
        self.df_unique = None
        if config == 'test':
            openml.config.start_using_configuration_for_example()

    def get_dataset(self):
        """
        Form a dataframe with the descriptions from all openml datasets
        :return: unique dataset descriptions with length min=50
        """
        dataset_list = datasets.list_datasets(output_format='dataframe', status='all')
        data_dict = defaultdict(list)
        for did in dataset_list['did']:
            try:
                data = datasets.get_dataset(did, download_data=False)
                if data.description is not None and data.name is not None:
                    data_dict['id'].append(did)
                    data_dict['name'].append(data.name)
                    data_dict['text'].append(data.description + " " + data.name + " ")
            except:
                # TODO: Exception type
                # For some reasons we get multiple exceptions apart from FileNotFound
                pass

        self.df = pd.DataFrame(data_dict)
        self.df.sort_values(by='id', inplace=True)
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
        # iterate through groups of same name and pick the row with longest description
        for name, group in grouped:
            idx = group["len"].idxmax()  # index of maximum len
            df_new = pd.concat([df_new, group.loc[[idx]]], ignore_index=True)
        return df_new



