from gensim.models import FastText
from gensim.models import Word2Vec

if __name__ == '__main__':
    model_ = FastText.load('fastText_Parsig.bin')
    print(model_.wv['dār'])
