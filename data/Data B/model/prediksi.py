from joblib import load
import pandas as pd
import os
import sys
sys.path.append('D:\github\python')
from Cek_typo import cek_typo as ct
from Normalisasi_KBBI import normalisasi_kbbi as nkbi
from modulku import praproses as pps
from modulku import StemNstopW as stm
from nltk.tokenize import TweetTokenizer
from essay_scoring.ES import scoring as sc
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
import re


def praproses(teks):
    a = tknzr.tokenize(teks)
    a = " ".join(a)
    a = pps.preprocessing(teks)
    a = pps.removePunc(a)
    a = ct.norm_typo(a)
    a = nkbi.norm_kbbi(a)
    a = stm.stemmer_kata(a)
    a = re.sub(' +', ' ',a)
    a = a.lstrip()
    return a


dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path+'/tfdf_B.joblib'
tfidf = load(filename)

filename = dir_path+'/SVM_B.joblib'
svm = load(filename)

def prediksi(x):
    if type(x) != list:
        return svm.predict(tfidf.transform([praproses(x)]).toarray())[0]
    elif type (x) == list:
        for ix, i in enumerate(x):
            new_x.append(praproses(i))
            
            if (ix+1)%100==0 and ix != 0:
                print(ix+1, end=" ")
            else:
                print(".", end="")
        print("|")
        print("<<Masuk Proses Prediksi>>")
        hasil_prediksi_list = svm.predict(tfidf.transform(new_x).toarray())
        dixt = {
            "prediksi":list(hasil_prediksi_list),
            "respon":x
        }
        return pd.DataFrame.from_dict(dixt)
    else:
        return "hanya menerima inputan bertipe list atau string"
        

def prediksi_list(x):
    new_x = list()
    #print("Praproses ->")
    for ix, i in enumerate(x):
        new_x.append(praproses(i))
        
        if (ix+1)%100==0 and ix != 0:
            print(ix+1, end=" ")
        else:
            print(".", end="")
    print("|")
    print("<<Masuk Proses Prediksi>>")
    hasil_prediksi_list = svm.predict(tfidf.transform(new_x).toarray())
    dixt = {
        "prediksi":list(hasil_prediksi_list),
        "respon":x
    }
    return pd.DataFrame.from_dict(dixt)