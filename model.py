import gensim
from gensim import corpora
import pandas as pd
from gensim.models import CoherenceModel
import numpy as np
import pyLDAvis.gensim
import pyLDAvis

df = pd.read_pickle("df_proc.pkl")

# Create a term dictionary of our corpus
docs = list(df["processed"].values)
dictionary = corpora.Dictionary(docs)

# Filter terms that occur in more than 50% of docs
dictionary.filter_extremes(no_above=0.5)

# Convert to document term matrix (corpus)
doc_term_mat = [dictionary.doc2bow(doc) for doc in docs]


def final_model():
    results = pd.read_csv("lda_tuning_results_1.csv")
    #results = results[results["topics"]== 6]
    best_params = results.sort_values(by="perplexity", ascending=True)
    beta = best_params["beta"].values[0]
    alpha = best_params["alpha"].values[0]
    if beta!="symmetric":
        beta = float(beta)
    if alpha != "symmetric" and alpha!="asymmetric":
        alpha = float(alpha)
    lda_model = gensim.models.LdaMulticore(corpus=doc_term_mat,
                                           workers=3,
                                           id2word=dictionary,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           per_word_topics=True,
                                           eta = beta,
                                           alpha=alpha,
                                           num_topics=int(best_params["topics"].values[0]))

    print(lda_model.get_document_topics(doc_term_mat))

    coherence_model_lda = CoherenceModel(model=lda_model,
                                         coherence='c_v',
                                         corpus=doc_term_mat,
                                         dictionary=dictionary,
                                         texts=df["processed"].values).get_coherence()
    perplexity = lda_model.log_perplexity(doc_term_mat)
    print("coherence final model ", coherence_model_lda)
    print("perplexity final model ", perplexity)
    topics = lda_model.print_topics()

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
                                           passes=10,
                                           per_word_topics=True)
    coherence_model_lda = CoherenceModel(model=lda_model,
                                         coherence='c_v',
                                         corpus=corpus,
                                         dictionary=id2word,
                                         texts = text)
    perplexity = lda_model.log_perplexity(corpus)
    return coherence_model_lda.get_coherence(), perplexity


def main():


    # LDA - This is our base model
    lda_model = gensim.models.LdaMulticore(corpus=doc_term_mat,
                                           id2word=dictionary,
                                           workers=3,
                                           num_topics=10,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
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

    # Hyper parameter tuning:
    topics_range = list(range(8, 10, 1))
    alpha_range = list(np.arange(0.01, 1, 0.3))
    alpha_range.append("symmetric")
    alpha_range.append("asymmetric")
    beta_range = list(np.arange(0.01, 1, 0.3))
    beta_range.append("symmetric")

    # Use 50% of data
    corpus_sets = [doc_term_mat]
    model_results ={
                    'alpha':[],
                    'beta':[],
                    'coherence':[],
                    'topics':[],
                     "perplexity":[]}

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


    # HDP = gensim.models.HdpModel
    # hdp_model = HDP(corpus=doc_term_mat,
    #                 id2word=dictionary)
    # hdp_topics = hdp_model.print_topics()
    # print("HDP topics:")
    # for topic in hdp_topics:
    #     print(topic)

if __name__ == "__main__":
    final_model()
    #main()