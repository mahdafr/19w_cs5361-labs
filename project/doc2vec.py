from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

# fixme: change to dataset location
dr = "D:\\Google Drive\\skool\\CS 5361\\datasets\\project\\"

""" Get the embeddings for each training sample """
def train(d, title, first_time=False):
    X, Y = d.train()
    x, y = d.test()

    if not first_time:
        print('Using ' +title+ ' model')
        model = Doc2Vec.load(dr+title+"d2v.model", )
        return vector(model, X), Y, vector(model, x), y

    train = [TaggedDocument(words=list(X), tags=list(Y))]
    np.save(dr + title + 'train', np.array(train))
    test = [TaggedDocument(words=list(x), tags=list(y))]
    np.save(dr + title + 'test', np.array(test))
    data = [TaggedDocument(words=list(np.append(X,x)),
                           tags=list(np.append(Y,y)))]
    np.save(dr + title + 'data', np.array(data))
    model = _model(title, data, epochs=2)
    return vector(model, X), Y, vector(model, x), y

""" Train the model for a Doc2Vec embedding of input data """
def _model(title, tag, epochs=100, v=10, alpha=0.025):
    # https://medium.com/@mishra.thedeepak/doc2vec-simple
    # -implementation-example-df2afbbfbad5
    model = Doc2Vec(size=v, alpha=alpha, min_alpha=0.00025,
                    min_count=1, dm=1)
    model.build_vocab(tag)

    # print('Training ' + title + 'doc2vec model')
    for epoch in range(epochs):
        model.train(tag,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        model.alpha -= 0.0002   # decrease the learning rate
        model.min_alpha = model.alpha   # fix to no decay
        # print('Completed epoch', epoch)

    print('Trained ' +title+ 'doc2vec model with epochs=',str(epochs),'vector_size=' + str(v))
    model.save(dr +title+ "d2v_v=" + str(v) + ".model")
    return model

""" Get the feature vector for the classifier """
def vector(model, doc):
    # https://towardsdatascience.com/multi-class-text-classification
    # -with-doc2vec-logistic-regression-9da9947b43f4
    # y, x = zip(*[(d.tags[0],
    #               model.infer_vector(d.words, steps=20))
    #              for d in doc])
    # return x, y
    ret = []
    for i in range(len(doc)):
        ret.append(model.infer_vector(list(doc[i])))
    return np.array(ret)
