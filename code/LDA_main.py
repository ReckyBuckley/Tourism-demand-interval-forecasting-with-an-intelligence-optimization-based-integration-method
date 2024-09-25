# -*- coding: utf-8 -*-

import pandas as pd
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

# Read EXCEL file containing Chinese comments
data = pd.read_excel("main\data\Social Reviews for Jiuzhaigou (text).xlsx",sheet_name=0)

data_set = [comment.split() for comment in data['Comment']]

# Creating dictionaries and corpus
dictionary = corpora.Dictionary(data_set)
corpus = [dictionary.doc2bow(comment) for comment in data_set]

#Calculating perplexity
def perplexity(num_topics):
    ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=30)
    print(ldamodel.print_topics(num_topics=num_topics, num_words=10))
    print(ldamodel.log_perplexity(corpus))
    return ldamodel.log_perplexity(corpus)
 
#Calculating coherence
def coherence(num_topics):
    ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=30,random_state = 1)
    print(ldamodel.print_topics(num_topics=num_topics, num_words=10))
    ldacm = CoherenceModel(model=ldamodel, texts=data_set, dictionary=dictionary, coherence='c_v')
    print(ldacm.get_coherence())
    return ldacm.get_coherence()
 
# Plotting a line graph of the coherence
x = range(1,10)
y=[]
for i in x:
    print(f"Processing {i} topics...")
    y.append(coherence(i))

plt.plot(x, y)
plt.xlabel('Topic number')
plt.ylabel('coherence')
plt.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
plt.title('Topic number-coherence')
plt.show()

y.to_excel("main\forecast results\Coherence for LDA.xlsx")

# Running the LDA model
optimal_idx = y.index(max(y))
optimal_num_topics = x[optimal_idx]
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=optimal_num_topics)

# Output and save the probability of each Chinese comment under each topic
topic_probabilities = []
for comment in tqdm(data_set, desc="Processing comments"):
    # Only probability values are retained
    prob_only = [prob for _, prob in lda_model.get_document_topics(dictionary.doc2bow(comment), minimum_probability=0)]
    topic_probabilities.append(prob_only)
    
topic_probabilities_df = pd.DataFrame(topic_probabilities)
topic_probabilities_df.fillna(0, inplace=True)

topic_probabilities_df.to_excel("main\Social Reviews\LDA result for Jiuzhaigou.xlsx")
