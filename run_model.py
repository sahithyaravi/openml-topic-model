import pandas as pd
from src.model import Model


def download_entire_dataset():
    from src.getdata import Dataset
    d = Dataset()
    df_raw = d.get_dataset()
    df_raw.to_csv('openml_dataset.csv')


def process_dataset():
    from src.preprocess import Process
    df_raw = pd.read_csv("data/openml_dataset.csv")
    processor = Process()
    df_proc = processor.get_processed_data(df_raw)
    df_proc.to_csv('data/openml_dataset_processed.csv')
    df_proc.to_pickle('data/openml_dataset_processed.pkl')


if __name__ == "__main__":
    df = pd.read_pickle('data/openml_dataset_processed.pkl')
    documents = df["processed"].values
    model = Model()
    print("calling train test split")
    model.train_test_split(documents)
    print("calling base model")
    base_lda = model.base_model(num_topics=11, alpha='auto', eta='auto')
    print(base_lda.eta, base_lda.alpha)
    # # print("Calling grid search")
    # # model.grid_search()
    # final_lda = model.final_model(path='results/noun_filter/lda_tuning_results.csv')
    # print("finalize and plot word cloud")
    # from src.utils import plot_word_cloud
    # plot_word_cloud(final_lda)
