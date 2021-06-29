import fasttext
import time

def train(inp = "wiki.he.text",out_model = "wiki.he.fasttext.model",
          alg = "CBOW"):

    start = time.time()

    model = fasttext.train_unsupervised(input=inp, model=alg)
    print(model.words) # list of words in dictionary

    print(time.time()-start)
          
    model.save(out_model)



def getModel(model = "wiki.he.fasttext.model.bin"):

    model = fasttext.load_model(model)

    return model
