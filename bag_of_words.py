from sklearn.feature_extraction.text import CountVectorizer


def BagOfWords(df):
    # Векторизуем простым способом BagOfWords
    count = CountVectorizer(analyzer='word')
    bag_of_words = count.fit_transform(df['review'])
    features = bag_of_words.toarray()
    return features
