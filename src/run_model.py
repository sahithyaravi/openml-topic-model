from src.preprocess import Process
from src.model import Model

if __name__ == '__main__':
    p = Process()
    df = p.get_processed_data(cache=True)
    documents = list(df["processed"]. values)
    m = Model()
    m.train_test_split(documents)
    #m.base_model(num_topics=15)
    m.grid_search()

