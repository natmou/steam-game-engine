import gensim
import pandas as pd
import cPickle as pickle
import numpy as np
import os
from gensim.models.keyedvectors import KeyedVectors
from flask import Flask, render_template, request, jsonify

app_path = '/Users/nmouterde/Documents/projectsPython/steam-game-engine/'

def search_game(search_sentences):
    words = search_sentences.split()
    search_vec = np.zeros(200)
    counter_w = 0
    for w in words:
        if w in word_model:
            search_vec = search_vec + word_model[w]
            counter_w+=1
        search_vec = search_vec/counter_w
    results = game_model.most_similar([search_vec], topn=10)
    df_results = pd.DataFrame.from_records(results, columns=['id', 'similarity'])
    df_results['id'] = df_results.id.astype(np.int64)
    df_results = pd.merge(df_results, data, on='id')
    return df_results.to_json(orient='index')

word_model = KeyedVectors.load_word2vec_format(os.path.join(app_path, 'models/glove_w2v_200'))
game_model = KeyedVectors.load_word2vec_format(os.path.join(app_path, 'models/gamevectors_w2c'))

data = pickle.load(open(os.path.join(app_path, 'data_processed/tf_idf_data.p'), 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template(os.path.join(app_path, 'index.html'))

@app.route('/get_search')
def get_search():
    search = request.args.get('search', '', type=str)
    search = search.replace('%20',' ')
    return jsonify(result=search_game(search)) 

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )