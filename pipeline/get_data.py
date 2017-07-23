import numpy as np
import pandas as pd
import requests
import json
import feather
import os
import cPickle as pickle
import gensim
from time import sleep
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

steam_id_url = 'http://api.steampowered.com/ISteamApps/GetAppList/v2'
steam_details_api = 'http://store.steampowered.com/api/appdetails/?appids='
ratings_api = 'http://steamspy.com/api.php?request=appdetails&appid='
app_path = '/Users/nmouterde/Documents/projectsPython/steam-game-engine/'

def get_data():
    steam_ids = requests.get(steam_id_url)
    steam_ids = json.loads(steam_ids.text)

    df_steam_ids = pd.DataFrame.from_dict(steam_ids['applist']['apps'])

    games = []
    for id_app in df_steam_ids.appid.values:
        api_call = requests.get(steam_details_api + str(id_app))
        rates_call = requests.get(ratings_api + str(id_app))
        rates = json.loads(rates_call.text)
        app_desc = json.loads(api_call.text)
        game = app_desc[app_desc.keys()[0]]
        if game['success']:
            game_info = game['data']
            if game_info['type'] == "game":
                #Treating special categories
                if 'categories' in game_info:
                    cats = [cat['description'] for cat in game_info['categories']]
                    cats = ' '.join(cats)
                else:
                    cats = ' '

                if 'recommendations' not in game_info:
                    rec = 0
                else:
                    rec = game_info['recommendations']['total']

                if 'developers' in game_info:
                    dev = game_info['developers'][0]
                else:
                    dev = ' '

                if 'genres' in game_info:
                    genres = [genre['description'] for genre in game_info['genres']]
                    genres = ' '.join(genres)
                else:
                    genre = ' '

                new_game = [game_info['steam_appid'], game_info['name'],
                              game_info['detailed_description'],
                              game_info['short_description'], 
                              game_info['about_the_game'],
                              dev,
                              game_info['publishers'][0],
                              cats, genres,
                              rates['score_rank'],
                              rec]
                
                games.append(new_game)
                
    df_cols = ['id', 'name',
               'f_desc', 's_desc','about',
               'dev' , 'pub', 'cat',
               'genres', 'score', 'nb_rec']

    df_info = pd.DataFrame.from_records(games, columns=df_cols)  
    df_info = df_info.drop_duplicates()  
    pickle.dump(df_info, open(os.path.join(app_path, 'data_processed/scrap_df.f'), 'wb'))

def clean_text():
    tkn = RegexpTokenizer(r'\w+')
    stoplist = stopwords.words('english')

    data = pickle.load(open(os.path.join(app_path, 'data_processed/scrap_df.f'), 'rb'))

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

def game_vector():

    def construct_vector(row):
        coeff = 0
        vector = np.zeros(200)
        for w in row['cat1']:
            if w in model:
                vector = vector + model[w]*1.5
                coeff += 1.5
        for w in row['cat2']:
            if w in model:
                vector = vector + model[w]
                coeff += 1
        for w in row['cat3']:
            if (w in model) and (w in names_tfidf):
                idx_tfidf = np.where(names_tfidf == w)[0][0]
                tfidf_score = row['tfidf_score'][idx_tfidf]
                c = tfidf_score*0.5
                coeff+=c
                vector = vector + model[w]*c
            
        vector = vector/coeff
        return vector.tolist()

    data = pickle.load(open(os.path.join(app_path, 'data_processed/tf_idf_data.p'), 'rb'))
    names_tfidf = pickle.load(open(os.path.join(app_path, 'data_processed/names_tf_idf.p'), 'rb'))
    names_tfidf = np.array(names_tfidf)

    glove2word2vec(os.path.join(app_path, 'models/glove.twitter.27B.200d.txt'),
                            os.path.join(app_path, 'models/glove_w2v_200'))
    model = KeyedVectors.load_word2vec_format(os.path.join(app_path, 'models/glove_w2v_200'))

    data['vec'] = data.apply(lambda x: construct_vector(x), axis=1)
    pickle.dump(data, open(os.path.join(app_path, 'data_processed/data_vectors.p'), 'wb'))

    write_line = data.apply(lambda x: str(x['id']) + ' ' + 
                        str(x['vec']).replace(',','')
                        .replace('[', '')
                        .replace(']', ''),
                        axis=1)

    fichier = open(os.path.join(app_path, "models/gamevectors"), "w") 
    for line in write_line.values:
        fichier.write(line+'\n')
    fichier.close() 

    glove2word2vec(os.path.join(app_path, "models/gamevectors"),
               os.path.join(app_path, "models/gamevectors_w2c"))

#get_data()
clean_text()
tf_idf_calculation()
game_vector()

