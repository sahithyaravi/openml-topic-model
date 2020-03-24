import os
import sys
import pandas as pd
import numpy as np

import gensim
from gensim.test.utils import datapath
from gensim.models import CoherenceModel
from gensim import corpora

import pyLDAvis.gensim
import pyLDAvis
from src.utils import compute_coherence_score

sys.path.append(os.getcwd())


class Model:
    def __init__(self):
        self.doc_term_mat_train = None
        self. doc_term_mat_test = None
        self.lda_model = None
        self.dictionary = None
        self.grid_search_results = None
        self.docs_train = None
        self.docs_test = None
        self.eta = None

    def train_test_split(self, docs):
        self.docs_train = docs
        self.docs_test = docs[5000:]
        self.dictionary = corpora.Dictionary(self.docs_train)

        # # Filter terms that occur in more than 50% of docs
        # self.dictionary.filter_extremes(no_above=0.7)

        # Convert to document term matrix (corpus)
        self.doc_term_mat_train = [self.dictionary.doc2bow(doc) for doc in self.docs_train]
        self.doc_term_mat_test = [self.dictionary.doc2bow(doc) for doc in self.docs_test]

    def set_priors(self, eta, topic, words, p=.8):
        """ for list of words set p(topic)=p
        eta is topic*word matrix with default p=1/topics
        """
        word2id = self.dictionary.token2id
        word_indexes = [word2id[w] for w in words]
        self.eta[topic, word_indexes] *= 2

    def get_eta(self, num_topics):
        self. eta = np.full((num_topics, len(self.dictionary)), 1 / (1 * num_topics))
        # print(self.eta)
        self.set_priors(self.eta, 0,
                        ['gene', 'oncology', 'tumor'])
        self.set_priors(self.eta, 1,
                        ['software', 'validation', 'code'])
        self.set_priors(self.eta, 2,
                        ['image', 'digit', 'pixel'])
        self.set_priors(self.eta, 3,
                        ['bid', 'forex', 'price', 'trade'])
        self.set_priors(self.eta, 4,
                        ['multivariate'])
        self.set_priors(self.eta, 5,
                        ['friedman'])
        self.set_priors(self.eta, 6,
                        ['drug', 'molecule'])
        self.set_priors(self.eta, 7,
                        ['biology', 'yeast', 'patient', 'disease'])
        return self.eta

    def base_model(self, num_topics=8):
        # LDA - This is our base model

        eta = self.get_eta(num_topics)
        lda_model = gensim.models.LdaMulticore(corpus=self.doc_term_mat_train,
                                               id2word=self.dictionary,
                                               alpha='asymmetric',
                                               eta=eta, #'auto',
                                               workers=3,
                                               chunksize=100,
                                               num_topics=num_topics,
                                               random_state=200,
                                               passes=1000,
                                               per_word_topics=True)

        topics = lda_model.print_topics()

        print("LDA topics for base model:")
        for topic in topics:
            print(topic)

        # Compute Coherence Score for base model
        coherence_model_lda = CoherenceModel(model=lda_model,
                                             corpus=self.doc_term_mat_train,
                                             texts=self.docs_train,
                                             dictionary=self.dictionary,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        # Visualize the topics
        vis_prepared = pyLDAvis.gensim.prepare(lda_model, self.doc_term_mat_train, self.dictionary)
        pyLDAvis.save_html(vis_prepared, "base.html")
        print('\nCoherence Score: ', coherence_lda)
        return lda_model

    def grid_search(self):
        # alpha = document-topic density
        # With a higher alpha, documents are made up of more topics, and with lower alpha, documents
        # contain fewer topics.
        # Beta represents =  topic-word density -
        # with a high beta,topics are made up of most of the words in the corpus,
        # and with a low beta they consist of few words.

        # Hyper parameter tuning:
        topics_range = list(range(7, 20, 1))
        alpha_range = list(np.arange(0.01, 1, 0.3))
        alpha_range.append("symmetric")
        alpha_range.append("asymmetric")
        alpha_range.append("auto")
        beta_range = list(np.arange(0.01, 1, 0.3))
        beta_range.append("symmetric")
        beta_range.append("auto")
        eta = self.get_eta(8)
        beta_range.append(eta)
        # Use 50% of data
        corpus_sets = [self.doc_term_mat_train]
        model_results = {
            'alpha': [],
            'beta': [],
            'coherence': [],
            'topics': [],
            "perplexity": []}

        for corpus in corpus_sets:
            for topic in topics_range:
                for alpha in alpha_range:
                    for beta in beta_range:
                        cv, p = compute_coherence_score(corpus, self.dictionary, topic,
                                                        alpha, beta, self.docs_train,
                                                        self.doc_term_mat_test)

                        model_results['topics'].append(topic)
                        model_results['alpha'].append(alpha)
                        model_results['beta'].append(beta)
                        model_results['coherence'].append(cv)
                        model_results['perplexity'].append(p)
        self.grid_search_results = pd.DataFrame(model_results)
        self.grid_search_results.to_csv('lda_tuning_results.csv', index=False)
        print("saved to results, done")

    def final_model(self):
        results = pd.read_csv("lda_tuning_results.csv")
        # results = results[results["topics"]== 6]
        more_topics = results[results["topics"] > 5]
        best_params = more_topics.sort_values(by="coherence", ascending=False)
        # best_params = best_params[best_params["perplexity"] > -7]
        beta = best_params["beta"].values[0]
        alpha = best_params["alpha"].values[0]
        print(best_params["coherence"].values[0])
        num_topics = int(best_params["topics"].values[0])
        print("topics ", num_topics)
        if beta != "symmetric" and beta != 'auto':
            beta = float(beta)
        if alpha != "symmetric" and alpha != "asymmetric" and alpha != "auto":
            alpha = float(alpha)
        lda_model = gensim.models.LdaModel(corpus=self.doc_term_mat_train,
                                           #workers=3,
                                           id2word=self.dictionary,
                                           random_state=100,
                                           chunksize=100,
                                           passes=200,
                                           iterations=1000,
                                           per_word_topics=True,
                                           eta=beta,
                                           alpha=alpha,
                                           num_topics=num_topics)

        coherence_model_lda = CoherenceModel(model=lda_model,
                                             coherence='c_v',
                                             corpus=self.doc_term_mat_train,
                                             dictionary=self.dictionary,
                                             texts=self.docs_train).get_coherence()
        perplexity = lda_model.log_perplexity(self.doc_term_mat_test)
        print("coherence final model ", coherence_model_lda)
        print("perplexity final model ", perplexity)
        topics = lda_model.show_topics(num_words=25)
        temp_path = datapath("model")
        lda_model.save(temp_path)

        for topic in topics:
            print(topic)

        # Visualize the topics
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, self.doc_term_mat_train, self.dictionary)
        pyLDAvis.save_html(LDAvis_prepared, "out.html")
        lda_model.update(self.doc_term_mat_test)
        return lda_model

    def predict_unseen_data_topic(self):
        unseen_doc = self.doc_term_mat_test[0]
        temp_file = datapath("model")
        lda = gensim.models.LdaMulticore.load(temp_file)
        lda.update(self.doc_term_mat_test) # Update the model by incrementally training on the new corpus
        vector = lda[unseen_doc]  # get topic probability distribution for a new document        print(vector)

    def save_all_topics(self, docs):
        doc_tops = []
        temp_file = datapath("model")
        lda_model = gensim.models.LdaMulticore.load(temp_file)
        topics = lda_model.show_topics(num_words=25)

        print(topics)
        for doc in docs:
            probs = dict(
                lda_model.get_document_topics(self.dictionary.doc2bow(doc)))
            doc_tops.append(probs)
        #df["topics"] = doc_tops
        # map_dict = {0: "other", 1:"price", 2:"bio", 3:"cv"}
        # df["topic_str"] = df["topics"].map(map_dict)
        #df.to_csv("resultdf.csv")



