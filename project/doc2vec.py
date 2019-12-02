from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

# fixme: change to dataset location
d = "D:\\Google Drive\\skool\\CS 5361\\datasets\\project\\"

""" Get the embeddings for each training sample """
def train(d, title, first_time=False):
    if not first_time:
        return Doc2Vec.load(d+title+"d2v.model")

    X, Y = d.train()
    x, y = d.test()
    data = [TaggedDocument(words=_d,
                           tags=list(np.append(Y,y)))
            for _d in enumerate(np.append(X,x))]
    model = _model(title, data, epochs=5)
    T = [TaggedDocument(words=_d,
                        tags=list(Y)) for _d in enumerate(X)]
    t = [TaggedDocument(words=_d,
                        tags=list(y)) for _d in enumerate(x)]
    return vector(model, T), vector(model, t)

""" Train the model for a Doc2Vec embedding of input data """
def _model(title, tag, epochs=100, v=20, alpha=0.025):
    # https://medium.com/@mishra.thedeepak/doc2vec-simple
    # -implementation-example-df2afbbfbad5
    model = Doc2Vec(size=v, alpha=alpha, min_alpha=0.00025,
                    min_count=1, dm=1)
    model.build_vocab(tag)

    for epoch in range(epochs):
        model.train(tag,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate, but fix to no decay
        model.alpha -= 0.0002
        model.min_alpha = model.alpha
        print('Completed epoch', epoch)

    model.save(d + title + "d2v.model")
    return model

""" Get the feature vector for the classifier """
def vector(model, doc):
    return zip(*[(d.tags[0], model.infer_vector(d.words, steps=20)) for d in doc.values])
