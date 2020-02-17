import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import FreqDist
import plotly
import plotly.graph_objs as go
from src.getdata import Dataset
from src.preprocess_functions import *


class Process:
    def __init__(self):
        nltk.download('stopwords')
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(["example", "experiment", "sample", "problem", "input", "output", "set", "task", "study",
                                "training", "prediction", "model", "test", "train",
                                "author", "source", "https", "uci", "edu", "citation", "html", "policy", "datum", "please",
                                "title", "dataset", "feature", "attribute", "attributes", "row", "column", "image", "file",
                                "pixel",
                                "description", "cite", "publication", "result", "distribution", "point",
                                "nominal", "enum", "string", "categorical", "number", "continuous", "numeric", "variable",
                                "instance", "set", "classtype", "none", "note", "inf", "information", "type", "data",
                                "target", "class", "positive", "negative", "value",
                                "time", "date", "year",
                                "imputation", "classification", "regression",
                                "colinearity", "degree", "average",
                                "unknown", "several", "version", "original",
                                "name", "project", "program", "paper", "thesis", "database", "format"
                                ])
        self.nlp = spacy.load('en')  # python -m spacy download en using admin on conda prompt

    def get_processed_data(self, cache: bool):
        dataset = Dataset()
        df = dataset.get_openml_data(cache)
        df["text"] = [text.lower() for text in df["text"]]
        df = remove_author_info(df)
        pd.set_option('display.expand_frame_repr', False)
        df = remove_special_chars(df)
        print(df.head())

        df['processed'] = [self.lemmetize(doc) for doc in df["text"]]
        df["processed"] = df['processed'].apply(lambda x: " ".join([word for word in x.split()
                                                                    if len(word) > 2]))
        df['processed'] = remove_stop_words(df['processed'], self.stop_words)
        self.plot_frequency_words(df['processed'])

        # Split to list of words
        processed_output = []
        for doc in df["processed"]:
            processed_output.append(doc.split())
        df["processed"] = get_bigrams(processed_output)
        df["title1"] = df["name"].str.split()
        df["processed"] = df["title1"] + df["processed"]
        df.to_pickle("df_proc.pkl")
        df.to_csv("df_proc.csv")
        return df

    def lemmetize(self, doc):
        sents = self.nlp(doc)
        doc_new = []
        for token in sents:
            if token.pos_ in ['NOUN']:
                doc_new.append(token.lemma_)
        return " ".join(doc_new)

    def plot_frequency_words(self, col):
        joined_words = " ".join([doc for doc in col])
        all_words = joined_words.split()
        fdist = FreqDist(all_words)
        df = (pd.DataFrame.from_dict(fdist, orient="index"))
        df.reset_index(inplace=True)
        df.columns = ["words", "count"]
        df.sort_values(by="count", inplace=True, ascending=False)
        data = [go.Bar(x=df["words"][:20], y=df["count"][:20])]
        fig = go.Figure(data)
        plotly.offline.plot(fig)


p = Process()
p.get_processed_data(cache=True)
