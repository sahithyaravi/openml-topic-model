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
import matplotlib.pyplot as plt
from gensim.test.utils import datapath
from wordcloud import WordCloud, STOPWORDS
#from .preprocess import *
# = pd.read_pickle("df_unique.pkl")

df = pd.read_pickle("df_proc.pkl")



# Create a term dictionary of our corpus
docs = list(df["processed"].values)
docs_train = docs[:2000]
docs_test =  docs[2000:]
dictionary = corpora.Dictionary(docs_train)


# Filter terms that occur in more than 50% of docs
dictionary.filter_extremes(no_above=0.5)

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
    temp_file = datapath("model")
    lda_model = gensim.models.LdaMulticore.load(temp_file)
    topics = lda_model.show_topics(num_words=25)

    print(topics)
    for doc in docs:
        probs = dict(
            lda_model.get_document_topics(dictionary.doc2bow(doc)))
        doc_tops.append(probs)
    df["topics"] = doc_tops
    # map_dict = {0: "other", 1:"price", 2:"bio", 3:"cv"}
    # df["topic_str"] = df["topics"].map(map_dict)
    df.to_csv("resultdf.csv")

def final_model():
    results = pd.read_csv("lda_tuning_results.csv")
    #results = results[results["topics"]== 6]
    best_params = results.sort_values(by="coherence", ascending=False)
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
    topics = lda_model.show_topics(num_words=25)
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
                                           random_state=200,
                                           passes=200,
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
    # Visualize the topics
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, doc_term_mat, dictionary)
    pyLDAvis.save_html(LDAvis_prepared, "base.html")
    print('\nCoherence Score: ', coherence_lda)


def hyper_parameter_find():
    # Hyper parameter tuning:
    topics_range = list(range(5, 20, 1))
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


def plot_word_cloud():
    tempfile = datapath("model")

    wc = WordCloud(
        # background_color="white",
        # max_words=2000,
        # width=1024,
        # height=720,
        # stopwords=stop_words
    )

    lda = gensim.models.LdaMulticore.load(tempfile)
    fig, axes = plt.subplots(2, 4, figsize=(20, 20), sharex=True, sharey=True)
    fig.delaxes(axes[1, 3])
    for t, ax in enumerate(axes.flatten()):
        if t!=lda.num_topics:
            #plt.figure()
            fig.add_subplot(ax)
            top_words = dict(lda.show_topic(t, 200))
            print("top_words\n", top_words)
            plt.gca().imshow(wc.fit_words(top_words))
            plt.gca().axis("off")
            #plt.gca().set_title("Topic #"+str(t))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
   #base_model()
   hyper_parameter_find()
   # final_model()
   # predict_unseen_data_topic()
   # save_all_topics()
   # plot_word_cloud()
   # result = pd.read_csv("resultdf.csv")
   # pd.set_option("display.max_colwidth",-1)
   # pd.set_option("display.max_rows",500)
   # # print(result[result["topics"]==4]["text"])