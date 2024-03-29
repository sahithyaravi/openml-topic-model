import os
import shutil
import sys
import pandas as pd
import numpy as np

import gensim
from gensim.test.utils import datapath
from gensim.models import CoherenceModel
from gensim import corpora

import pyLDAvis.gensim
import pyLDAvis
from .utils import compute_coherence_score

sys.path.append(os.getcwd())
from gensim.models.wrappers import LdaMallet, ldamallet


class Model:
    def __init__(self):

        self.doc_term_mat_train = None
        self.doc_term_mat_test = None
        self.lda_model = None
        self.dictionary = None
        self.grid_search_results = None
        self.docs_train = None
        self.docs_test = None
        self.eta = None
        self.df = None

    def train_test_split(self, docs, filter=0.5, df=None):
        # docs = [d.split() for d in docs]
        self.docs_train = docs
        self.df = df
        self.docs_test = docs[2000:]
        self.dictionary = corpora.Dictionary(self.docs_train)

        # # Filter terms that occur in more than 50% of docs
        self.dictionary.filter_extremes(no_above=filter
                                        )

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

    def base_model(self, num_topics=8, eta='auto', alpha='auto'):
        # LDA - This is our base model
        if eta == 'seed':
            eta = self.get_eta(num_topics)
        lda_model = gensim.models.LdaModel(corpus=self.doc_term_mat_train,
                                           id2word=self.dictionary,
                                           alpha=alpha,
                                           eta=eta,
                                           # workers=3,
                                           # chunksize=100,
                                           iterations=1000,
                                           num_topics=num_topics,
                                           random_state=150,
                                           passes=100,
                                        )
        # path_to_mallet_binary = r'C:\mallet\bin\mallet'
        # lda_model = gensim.models.wrappers.LdaMallet(path_to_mallet_binary, corpus=self.doc_term_mat_train,
        #                                              id2word=self.dictionary,
        #                                              workers=4,
        #                                              iterations=1000,
        #                                              alpha=5,
        #                                              num_topics=num_topics,
        #                                              )
        # lda_model = ldamallet.malletmodel2ldamodel(lda_model)
        #
        topics = lda_model.print_topics()


        print("LDA topics for base model:")
        print("alpha", lda_model.alpha)
        print("beta", lda_model.eta)
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

    def grid_search(self, result_folder =""):
        # alpha = document-topic density
        # With a higher alpha, documents are made up of more topics, and with lower alpha, documents
        # contain fewer topics.
        # Beta represents =  topic-word density -
        # with a high beta,topics are made up of most of the words in the corpus,
        # and with a low beta they consist of few words.

        # Hyper parameter tuning:
        topics_range = [10, 12, 15]
        alpha_range = list(np.arange(0.01, 0.21, 0.05))
        alpha_range.append("symmetric")
        alpha_range.append("asymmetric")
        alpha_range.append("auto")
        beta_range = [0.1, "symmetric", "auto"]  # list(np.arange(0.01, 0.21, 0.05))
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
                        if True:
                            print("Grid search", topic, alpha, beta)
                            # beta = self.get_eta(topic)
                            cv, p, _ = compute_coherence_score(corpus,
                                                               self.dictionary,
                                                               topic,
                                                               alpha,
                                                               beta,
                                                               self.docs_train,
                                                               self.doc_term_mat_test)

                            model_results['topics'].append(topic)
                            model_results['alpha'].append(alpha)
                            model_results['beta'].append(beta)
                            model_results['coherence'].append(cv)
                            model_results['perplexity'].append(p)
        self.grid_search_results = pd.DataFrame(model_results)
        self.grid_search_results.to_csv(result_folder + 'lda_tuning_results.csv', index=False)
        print("saved to results, done")

    def final_model(self, path='lda_tuning_results.csv', n=0, result_folder=""):
        results = pd.read_csv(path)
        best_params = results
        # best_params = results[results["alpha"].isin(["symmetric", "asymmetric"])]
        # best_params = best_params[best_params["perplexity"] > -7]
        best_params = best_params.sort_values(by="coherence", ascending=False)
        print(best_params.head(10))

        beta = best_params["beta"].values[n]
        alpha = best_params["alpha"].values[n]
        print(best_params["coherence"].values[n])
        num_topics = int(best_params["topics"].values[n])
        print("topics ", num_topics)
        if beta != "symmetric" and beta != 'auto':
            beta = float(beta)
        if alpha != "symmetric" and alpha != "asymmetric" and alpha != "auto":
            alpha = float(alpha)
        print(alpha, beta, num_topics)

        cv, p, lda_model = compute_coherence_score(self.doc_term_mat_train,
                                                   self.dictionary,
                                                   num_topics,
                                                   alpha,
                                                   beta,
                                                   self.docs_train,
                                                   self.doc_term_mat_test)
        print("coherence final model ", cv)
        print("perplexity final model ", p)
        temp_path = datapath("model")
        lda_model.save(temp_path)

        topics = []
        for t in range(num_topics):
            top_words = lda_model.show_topic(t, 10)
            only_words = [x[0] for x in top_words]
            topics.append(only_words)
            print(only_words)


        results_dict = {
            "preprocessing": "noun",
            "eta": lda_model.eta,
            "alpha": lda_model.alpha,
            "coherence": cv,
            "perplexity": p,
            "topwords": topics
        }

        # Visualize the topics
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, self.doc_term_mat_train, self.dictionary)
        pyLDAvis.save_html(LDAvis_prepared, "out.html")
        shutil.move('out.html', result_folder+'out.html')
        lda_model.update(self.doc_term_mat_test)
        return lda_model, results_dict

    def predict_unseen_data_topic(self):
        unseen_doc = self.doc_term_mat_test[0]
        temp_file = datapath("model")
        lda = gensim.models.LdaMulticore.load(temp_file)
        lda.update(self.doc_term_mat_test) # Update the model by incrementally training on the new corpus
        vector = lda[unseen_doc]  # get topic probability distribution for a new document        print(vector)

    def save_all_topics(self, lda_model, folder_path , topwords):
        docs = self.docs_train
        best_topics = []
        topic_probabilities = []
        topic_info = []


        for doc in docs:
            probs = dict(
                lda_model.get_document_topics(self.dictionary.doc2bow(doc)))
            topic_probabilities.append(probs)
            t = max(probs, key=probs.get)
            p = max(probs.values())
            topic_info.append(topwords[t])
            if p < 0.2:
                t = -1
            best_topics.append(t)
        self.df['distribution_topics'] = topic_probabilities
        self.df["best_topic"] = best_topics
        self.df["topic_info"] = topic_info
        # df["topic_str"] = df["topics"].map(map_dict)
        self.df.to_csv(folder_path+"resultdf.csv")



