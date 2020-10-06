import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import FreqDist
import plotly
import plotly.graph_objs as go
from src.preprocess_functions import *


class Process:
    def __init__(self, parts_of_speech=['NOUN', 'PROPN']):
        nltk.download('stopwords')
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(["example", "experiment", "sample", "problem", "input", "output",
                                "set", "task", "study",
                                "training", "prediction", "model", "test", "train",
                                "author", "source", "https", "uci", "edu",
                                "citation", "html", "policy", "datum",
                                "please", "arff", "title", "dataset", "feature",
                                "attribute", "attributes", "row", "column", "file",
                                "description", "cite", "publication", "result",
                                "distribution", "point",
                                "nominal", "enum", "string", "categorical",
                                "number", "continuous", "numeric",
                                "variable", "instance", "set", "classtype", "none", "note", "inf",
                                "information", "type", "data",
                                "target", "class", "positive", "negative", "value",
                                "time", "date", "year",
                                "imputation", "classification", "regression",
                                "colinearity", "degree", "average",
                                "unknown", "several", "version", "original", "unit",
                                "name", "project", "program",
                                "paper", "thesis", "database", "format"
                                ])
        # run 'python -m spacy download en' as admin on conda prompt
        self.nlp = spacy.load('en')
        self.parts_of_speech = parts_of_speech

    def get_processed_data(self, df):
        """

        :param df: data frame created using src.getdata.Dataset, contains "text" column
        :return: processed data frame with text processed and returned in "processed" column of data frame
        """

        df = remove_author_info(df)
        pd.set_option('display.expand_frame_repr', False)
        df = remove_special_chars(df)
        df = lower_case(df)

        # Lemmetize documents
        df['processed'] = [self.lemmetize(doc) for doc in df["text"]]
        # Remove words of length 0-2
        df["processed"] = df['processed'].apply(lambda x: " ".join([word for word in x.split()
                                                                    if len(word) > 3]))
        df['processed'] = remove_stop_words(df['processed'], self.stop_words)
        self.plot_frequency_words(df['processed'])
        # Split to list of words
        processed_output = []
        for doc in df["processed"]:
            processed_output.append(doc.split())
        df["processed"] = get_bigrams(processed_output)

        return df

    def lemmetize(self, doc):
        sents = self.nlp(doc)
        doc_new = []
        # https://spacy.io/usage/linguistic-features
        for token in sents:
            if token.pos_ in self.parts_of_speech:
                if token.ent_type_ not in ['PERSON', 'GPE', 'ORG', 'NORP']:
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

