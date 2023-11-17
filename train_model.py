import os
import re
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

# пакеты для стемминга, словари с stopwords
import nltk
from nltk.corpus import stopwords
import snowballstemmer

# разные векторизаторы
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# для выделения подвыборки
from sklearn.model_selection import train_test_split
# для вычисления оценок качества классификации
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# классификаторы
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from bag_of_words import BagOfWords


def TrainModel(trainFolder):
    # директория с файлами-отзывами
    df = pd.DataFrame(columns=['review', 'sentiment'])

    nltk.download('stopwords')
    stop_words = stopwords.words('russian')
    stemmer = snowballstemmer.stemmer('russian')

    # чтение обучающих данных
    path = trainFolder
    for directory in os.listdir(path):
        if os.path.isdir(path + directory):
            files = np.array(os.listdir(path + directory))
            for file in files:
                with open(os.path.join(path + directory + '/', file), encoding='utf-8') as f:
                    data = f.read().lower()
                    # удаляем цифры и дополнительные символы
                    data = re.sub(r'[0-9\'\"\»\«\(\)\…\–\-\—\,\.\№\:\;\?\!\/]', '', data).strip()
                    data = data.split()
                    # удаляем стоп-слова
                    tmp = [i for i in data if i not in stop_words]
                    # стемминг
                    tmp2 = stemmer.stemWords(tmp)
                    review = ' '.join(tmp2)
                    current_df = pd.DataFrame({'review': [review], 'sentiment': directory})
                    df = pd.concat([df, current_df], ignore_index=True)
    features = BagOfWords(df)
    target = df['sentiment']

    # разбиваем имеющуюся выборку на части
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    # обучение модели
    classifier = LinearSVC()
    with warnings.catch_warnings():
        # игнорируем ошибки
        warnings.filterwarnings("ignore")
        classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
