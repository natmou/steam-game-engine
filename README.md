# steam-game-engine
Search and recommendation engine based on apps steam description

This project is about creating a model to allow research of steam games thanks to sentence query.

I am trying to create vectors from word description of game using word2vec existing model and technique.

From there, I can compare game vectors and word vectors to perform a seach based on a sentence.

If you want to try the code, you will need to download the twitter pretrained glove model here : http://nlp.stanford.edu/data/glove.twitter.27B.zip.

If it is just for a short test I would recommend you to use the get_data function or script only on hundreds steam game ids. The requests on steam api are blocked after 200 and it takes some times to get every games.

