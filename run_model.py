import pandas as pd
# from src.preprocess import Process
from src.model import Model
# from src.utils import plot_word_cloud
"""
    from src.getdata import Dataset
    d = Dataset()
    df = d.get_dataset()
    df.to_csv('openml_dataset.csv')
"""

if __name__ == "__main__":

    # Step 1 process dataset
    # df = pd.read_csv("data/openml_dataset.csv")
    # processor = Process()
    # df = processor.get_processed_data(df)

    df = pd.read_pickle("df_proc.pkl")
    documents = list(df["processed"].values)
    m = Model()
    print("calling train test split")
    m.train_test_split(documents)
    print("calling base model")
    lda = m.base_model()
    print("Calling grid search")
    m.grid_search()
    # lda = m.final_model()
    # plot_word_cloud(lda)
