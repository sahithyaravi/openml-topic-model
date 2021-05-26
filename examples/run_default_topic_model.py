import pandas as pd
import shutil, os

from openml_topic.model import Model
from openml_topic.getdata import Dataset
from openml_topic.preprocess import Process
from config import *


def download_entire_dataset():
    d = Dataset()
    df_raw = d.get_dataset()
    df_raw.to_csv('../data/openml_dataset.csv')


def process_dataset(parts_of_speech):
    df_raw = pd.read_csv("../data/openml_dataset.csv")
    processor = Process(parts_of_speech=parts_of_speech)
    df_proc = processor.get_processed_data(df_raw)
    print("saving processed dataset")
    df_proc.to_csv('../data/openml_dataset_processed.csv')
    df_proc.to_pickle('../data/openml_dataset_processed.pkl')


def run():
    if DOWNLOAD_DATASET_AGAIN:
        download_entire_dataset()
    if PROCESS_DATASET:
        process_dataset(parts_of_speech=PROCESSING_PARTS_OF_SPEECH_INCLUDED)

    df = pd.read_pickle('../data/openml_dataset_processed.pkl')
    df.dropna(inplace=True)
    documents = df["processed"].values
    POS = "".join(PROCESSING_PARTS_OF_SPEECH_INCLUDED)
    FILTER = FILTER_WORDS
    result_folder = f"../results/{POS}_filter_{FILTER}/"
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    print("calling train test split")
    model = Model()
    model.train_test_split(documents, filter=FILTER, df=df)
    if GRID_SEARCH:
        print("Calling grid search")
        model.grid_search(result_folder)
    if FINAL_MODEL:
        final_lda, result_dict = model.final_model(n=0, path=result_folder+'lda_tuning_results.csv',
                                                   result_folder=result_folder)
        model.save_all_topics(lda_model=final_lda, folder_path=result_folder, topwords = result_dict['topwords'])

        result_dict['filter'] = FILTER
        result_dict['pos'] = POS
        if os.path.exists("../results/results_all_runs.pkl"):
            results = pd.read_pickle("../results/results_all_runs.pkl")
        else:
            results = pd.DataFrame()

        results.append(pd.DataFrame(result_dict.items()), ignore_index=True)
        print(result_dict)
        # print("finalize and plot word cloud")
        # from openml_topic.utils import plot_word_cloud
        # plot_word_cloud(final_lda, result_folder)


if __name__ == "__main__":
    run()

