from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import numpy as np

""" Get the embeddings for each training sample """
def train(d, title, first_time=False):
    X, Y = d.train()
    x, y = d.test()

    if not first_time:
        return Doc2Vec.load("d2v.model").docvecs
    # https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
    data = np.append(X,x)   # of all the samples
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
    max_epochs = 5  # 100
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(size=vec_size, alpha=alpha, min_alpha=0.00025,
                    min_count=1, dm=1)
    model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate, but fix to no decay
        model.alpha -= 0.0002
        model.min_alpha = model.alpha
    model.save(title+"d2v.model")
    model = Doc2Vec.load("d2v.model")
    return model.docvecs