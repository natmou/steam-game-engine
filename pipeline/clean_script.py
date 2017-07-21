import numpy as np
import pandas as pd
import requests
import json
import feather
import os
import cPickle as pickle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

app_path = '/Users/nmouterde/Documents/projectsPython/steam-game-engine/'

def clean_text():
    tkn = RegexpTokenizer(r'\w+')
    stoplist = stopwords.words('english')

    data = feather.read_dataframe(os.path.join(app_path + 'data_processed/scrap_df.f'))

    data['cat1'] = data['name'].str.lower()
    data['cat2'] = (data['cat'] + ' '
                          +data['genres']).str.lower()
    data['cat3'] = np.where(data['f_desc'] == data['about'],
                                  (data['f_desc'] + ' ' +
                                   data['about']).str.lower(),
                                  data['f_desc'].str.lower())

    cols = ['cat1', 'cat2', 'cat3']
    for col in cols:
        data[col] = data.apply(lambda x: tkn.tokenize(x[col]), axis=1)
        data[col] = data.apply(lambda x: [w for w in x[col] 
                                         if w not in stoplist and len(w)>1],
                               axis=1) 
        data[col+'_non_tk'] = data.apply(lambda x: ' '.join(x[col]), axis=1)

    pickle.dump(data, open(os.path.join(app_path, 'data_processed/clean_data.p'), 'wb'))

def tf_idf_calculation():
    data = pickle.load(open(os.path.join(app_path, 'data_processed/clean_data.p'), 'rb'))
    vect = TfidfVectorizer(sublinear_tf=True, max_df=1.0, analyzer='word', stop_words='english')

    X = vect.fit_transform(data['cat3_non_tk']).toarray()
    data['tfidf_score'] = X.tolist()

    name_corr = vect.get_feature_names()

    pickle.dump(data, open(os.path.join(app_path, 'data_processed/tf_idf_data.p'), 'wb'))
    pickle.dump(name_corr, open(os.path.join(app_path, 'data_processed/names_tf_idf.p'), 'wb'))

clean_text()
tf_idf_calculation()