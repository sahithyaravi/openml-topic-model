import pandas as pd
from src.model import Model


if __name__ == "__main__":
    df = pd.read_pickle("data/df_proc.pkl")
    documents = list(df["processed"]. values)
    m = Model()
    print("calling train test split")
    m.train_test_split(documents)
    print("calling base model")
    #m.base_model(num_topics=15)
    print("Calling grid search")
    m.grid_search()

