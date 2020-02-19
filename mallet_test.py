from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.wrappers import LdaMallet
import os
os.environ['MALLET_HOME'] = "C:\\Users\\s164255\\mallet-2.0.8\\mallet-2.0.8\\bin\\mallet"
path_to_mallet_binary = "C:\\Users\\s164255\\mallet-2.0.8\\mallet-2.0.8\\bin\\mallet"
model = LdaMallet(path_to_mallet_binary, corpus=common_corpus, num_topics=20, id2word=common_dictionary)
vector = model[common_corpus[0]]
