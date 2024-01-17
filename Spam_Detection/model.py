import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

data = pd.read_csv("./spam.csv",encoding='latin1')
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
data.dropna(inplace=True)

x = data['v2']
y = data['v1']
vect = CountVectorizer()

x_vect = vect.fit_transform(x)
x_df = pd.DataFrame(x_vect.toarray(), columns=vect.get_feature_names_out())


classifier = MultinomialNB()
classifier.fit(x_df, y)

joblib.dump(vect,"vectorizer.joblib")
joblib.dump(classifier, "spam_clf.joblib")