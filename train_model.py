import os
import re
import warnings

import numpy as np
import pandas as pd

# пакеты для стемминга, словари с stopwords
import nltk
from nltk.corpus import stopwords
import snowballstemmer

from sklearn.model_selection import train_test_split  # для выделения подвыборки
from sklearn.metrics import classification_report  # для вычисления оценок качества классификации

# классификаторы
from sklearn.svm import LinearSVC

from sklearn.preprocessing import normalize

from bag_of_words import BagOfWords


def TrainModel(trainFolder):
    # Директория с файлами-отзывами
    df = pd.DataFrame(columns=['review', 'sentiment'])

    nltk.download('stopwords')
    stop_words = stopwords.words('russian')
    stemmer = snowballstemmer.stemmer('russian')

    # Чтение обучающих данных
    path = trainFolder
    for directory in os.listdir(path):
        if os.path.isdir(path + directory):
            dirs = np.array(os.listdir(path + directory))

            np.random.shuffle(os.listdir(path + directory))
            files = np.random.choice(dirs, round(len(dirs)))

            for file in files:
                with open(os.path.join(path + directory + '/', file), encoding='utf-8') as f:
                    data = f.read().lower()
                    # Чистка содержимого
                    # data = re.sub(r'[\'\"\»\«\(\)\…\–\-\—\,\.\№\:\;\?\!\/]', '', data).strip()
                    # data = data.split()
                    # Удаление стоп-слов
                    # tmp = [i for i in data if i not in stop_words]
                    # Стемминг
                    # tmp2 = stemmer.stemWords(tmp)
                    # review = ' '.join(tmp2)
                    # Леминг

                    current_df = pd.DataFrame({'review': [data], 'sentiment': directory})
                    df = pd.concat([df, current_df], ignore_index=True)
    features = BagOfWords(df)
    target = df['sentiment']

    # разбиваем имеющуюся выборку на части и проводим нормализацию
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)
    X_train, X_test = normalize(X_train), normalize(X_test)

    # rus = RandomUnderSampler(random_state='RS')
    # X_under, y_under = rus.fit_sample(X_train, y_train)

    # обучение модели
    clf = LinearSVC()
    with warnings.catch_warnings():
        # игнорируем ошибки
        warnings.filterwarnings("ignore")
        clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
