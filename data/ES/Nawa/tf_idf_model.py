from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump


def tf_idf_bulid(list_, vocab=[], fitur=False, char = False):
    if fitur==True and char==True:
        # print("TT", end=".")
        if type(vocab)==list:
            vocab = " ".join(vocab)
        vocab = list(set(vocab))
        vocab = " ".join(vocab)
        vocab = list(set(vocab))
        if len(vocab)>= 1:
            vectorizer = TfidfVectorizer(vocabulary=vocab, analyzer='char')
            # dump(vectorizer.fit(list_), 'model_tdidf/tdfidf.joblib')
        else:
            vectorizer = TfidfVectorizer()
            # dump(vectorizer.fit(list_), 'model_tdidf/tdfidf.joblib')
        # return vectorizer.fit_transform([text1, text2])
    elif fitur==True and char==False:
        # print("TF", end=".")
        if len(vocab)>= 1:
            vectorizer = TfidfVectorizer(vocabulary=vocab)
            # dump(vectorizer.fit(list_), 'model_tdidf/tdfidf.joblib')
        else:
            vectorizer = TfidfVectorizer()
            # dump(vectorizer.fit(list_), 'model_tdidf/tdfidf.joblib')
        # return vectorizer.fit_transform([text1, text2])
    elif fitur==False and char==True:
        # print("FT", end=".")
        vectorizer = TfidfVectorizer(analyzer='char')
        # dump(vectorizer.fit(list_), 'model_tdidf/tdfidf.joblib')
        # return vectorizer.fit_transform([text1, text2])
    else:
        # print("FF")
        vectorizer = TfidfVectorizer()
        # dump(vectorizer.fit(list_), 'model_tdidf/tdfidf.joblib')
        # return vectorizer.fit_transform([text1, text2])
    
    save = vectorizer.fit(list_)
    dump(vectorizer.fit(list_), 'model_tdidf/tdfidf.joblib'))
    tfidf = vectorizer.fit_transform(list_)

def tf_idf_(list_, char = False):
    # fitur = list(set(fitur))
    if char == True:
        vectorizer = TfidfVectorizer(analyzer='char')
    else:
        vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(list_)