import os
import sys
sys.path.append(os.getcwd())

import gensim
from gensim import corpora
import pandas as pd
from gensim.models import CoherenceModel
import numpy as np
import pyLDAvis.gensim
import pyLDAvis
from gensim.test.utils import datapath
df_original = pd.read_pickle("df_unique.pkl")

df = pd.read_pickle("df_proc.pkl")


# Create a term dictionary of our corpus
docs = list(df["processed"].values)
docs_train = docs[:2000]
docs_test =  docs[2000:]
dictionary = corpora.Dictionary(docs_train)


# Filter terms that occur in more than 50% of docs
dictionary.filter_extremes(no_above=0.4)

# Convert to document term matrix (corpus)
doc_term_mat = [dictionary.doc2bow(doc) for doc in docs_train]
doc_term_mat_test = [dictionary.doc2bow(doc) for doc in docs_test]


def predict_unseen_data_topic():
    unseen_doc = doc_term_mat_test[0]
    temp_file = datapath("model")
    lda = gensim.models.LdaMulticore.load(temp_file)
    lda.update(doc_term_mat_test) # Update the model by incrementally training on the new corpus
    vector = lda[unseen_doc]  # get topic probability distribution for a new document
    print(vector)


def save_all_topics():
    docs = list(df["processed"].values)
    doc_tops = []
    unseen_doc = doc_term_mat_test[0]
    temp_file = datapath("model")
    lda_model = gensim.models.LdaMulticore.load(temp_file)
    for doc in docs:
        probs = dict(
            lda_model.get_document_topics(dictionary.doc2bow(doc)))
        t = max(probs, key=probs.get)
        doc_tops.append(t)
    df["topics"] = doc_tops
    map_dict = {0: "other", 1:"price", 2:"bio", 3:"cv"}
    df["topic_str"] = df["topics"].map(map_dict)
    df.to_csv("resultdf.csv")

def final_model():
    results = pd.read_csv("lda_tuning_results.csv")
    #results = results[results["topics"]== 10]
    best_params = results.sort_values(by="perplexity", ascending=False)
    #best_params = best_params[best_params["perplexity"] > -7]
    beta = best_params["beta"].values[0]
    alpha = best_params["alpha"].values[0]
    print(best_params["coherence"].values[0])
    num_topics = int(best_params["topics"].values[0])
    print("topics ", num_topics)
    if beta!="symmetric":
        beta = float(beta)
    if alpha != "symmetric" and alpha!="asymmetric":
        alpha = float(alpha)
    lda_model = gensim.models.LdaMulticore(corpus=doc_term_mat,
                                           workers=3,
                                           id2word=dictionary,
                                           random_state=100,
                                           chunksize=100,
                                           passes=100,
                                           per_word_topics=True,
                                           eta = beta,
                                           alpha=alpha,
                                           num_topics=num_topics)



    coherence_model_lda = CoherenceModel(model=lda_model,
                                         coherence='c_v',
                                         corpus=doc_term_mat,
                                         dictionary=dictionary,
                                         texts=df["processed"].values).get_coherence()
    perplexity = lda_model.log_perplexity(doc_term_mat_test)
    print("coherence final model ", coherence_model_lda)
    print("perplexity final model ", perplexity)
    topics = lda_model.print_topics()
    temp_path = datapath("model")
    lda_model.save(temp_path)


    for topic in topics:
        print(topic)


    # Visualize the topics
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, doc_term_mat, dictionary)
    pyLDAvis.save_html(LDAvis_prepared, "out.html")


def compute_coherence_score(corpus, id2word, num_topics, alpha, eta, text):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           workers=3,
                                           id2word=id2word,
                                           alpha=alpha,
                                           eta=eta,
                                           num_topics=num_topics,
                                           random_state=100,
                                           chunksize=100,
                                           passes=200,
                                           per_word_topics=True)
    coherence_model_lda = CoherenceModel(model=lda_model,
                                         coherence='c_v',
                                         corpus=corpus,
                                         dictionary=id2word,
                                         texts = text)
    perplexity = lda_model.log_perplexity(doc_term_mat_test)
    return coherence_model_lda.get_coherence(), perplexity


def base_model():


    # LDA - This is our base model
    lda_model = gensim.models.LdaMulticore(corpus=doc_term_mat,
                                           id2word=dictionary,
                                           workers=3,
                                           chunksize=100,
                                           num_topics=10,
                                           random_state=100,
                                           passes=100,
                                           per_word_topics=True)


    topics = lda_model.print_topics()
    print("LDA topics for base model:")
    for topic in topics:
        print(topic)

    # Compute Coherence Score for base model
    coherence_model_lda = CoherenceModel(model=lda_model,
                                         corpus=doc_term_mat,
                                         texts=df['processed'].values,
                                         dictionary=dictionary,
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)


def hyper_parameter_find():
    # Hyper parameter tuning:
    topics_range = list(range(5, 11, 1))
    alpha_range = list(np.arange(0.01, 1, 0.3))
    alpha_range.append("symmetric")
    alpha_range.append("asymmetric")
    beta_range = list(np.arange(0.01, 1, 0.3))
    beta_range.append("symmetric")

    # Use 50% of data
    corpus_sets = [doc_term_mat]
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
                    cv, p = compute_coherence_score(corpus, dictionary, topic,
                                                    alpha, beta, df["processed"].values)

                    model_results['topics'].append(topic)
                    model_results['alpha'].append(alpha)
                    model_results['beta'].append(beta)
                    model_results['coherence'].append(cv)
                    model_results['perplexity'].append(p)
    results = pd.DataFrame(model_results)
    results.to_csv('lda_tuning_results.csv', index=False)
    print("saved to results, done")


if __name__ == "__main__":
   # base_model()
   #hyper_parameter_find()
   final_model()
   #predict_unseen_data_topic()
   #save_all_topics()
   # result = pd.read_csv("resultdf.csv")
   # pd.set_option("display.max_colwidth",-1)
   # print(result[result["topic_str"]=="other"]["text"])