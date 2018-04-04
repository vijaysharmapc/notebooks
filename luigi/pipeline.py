import luigi
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle


class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows
        without valid geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0,  0.0) files.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='clean_data.csv')

    def requires(self):
        return None

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        airline_tweets = pd.read_csv(
            self.tweet_file, usecols=['airline_sentiment',  'tweet_coord'],
            encoding='utf-8')
        airline_tweets = airline_tweets.dropna()
        airline_tweets = airline_tweets.loc[airline_tweets['tweet_coord']
            != '[0.0, 0.0]']
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        airline_tweets['airline_sentiment'] = \
            airline_tweets['airline_sentiment'].map(sentiment_map)
        for ch in ['\[', '\]']:
            airline_tweets['tweet_coord'].replace(
               regex=True, inplace=True, to_replace=ch, value=r'')
        airline_tweets = airline_tweets.reset_index(drop=True)
        split_c = airline_tweets['tweet_coord'].apply(lambda x: x.split(','))
        airline_tweets['latitude'] = split_c.apply(lambda x: float(x[0]))
        airline_tweets['longitude'] = split_c.apply(lambda x: float(x[1]))
        airline_tweets.drop(['tweet_coord'], axis=1)
        airline_tweets.to_csv(self.output_file)


class TrainingDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid
        geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0,  0.0) files.
    """
    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')
    clean_data = luigi.Parameter(default='clean_data.csv')

    def requires(self):
        return CleanDataTask(tweet_file=self.tweet_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        cities = pd.read_csv(
            self.cities_file,  delimiter=",", header='infer',
            usecols=['name', 'latitude', 'longitude'], encoding='utf-8')
        cities = cities.dropna()
        cities['city_coord'] = cities[['latitude', 'longitude']]\
            .values.tolist()
        airline_tweets = pd.read_csv(
            self.clean_data,  delimiter=",", header='infer')

        #function to find the nearest city
        def closest_location(setA_lat, setA_lng, setB_lat, setB_lng):
            a_x = setA_lat.values
            a_y = setA_lng.values
            b_x = setB_lat.values
            b_y = setB_lng.values
            a = np.c_[a_x,  a_y]
            b = np.c_[b_x,  b_y]
            indx = cKDTree(b).query(a, k=1, p=2)[1]
            return pd.Series(
                b_x[indx]),  pd.Series(b_y[indx]), \
                pd.Series(cities['name'][indx])

        setA_lat = airline_tweets['latitude']
        setA_lng = airline_tweets['longitude']
        setB_lat = cities['latitude']
        setB_lng = cities['longitude']

        c_x, c_y, c_n = closest_location(
            setA_lat, setA_lng, setB_lat, setB_lng)
        c_n = c_n.reset_index(drop=True)

        airline_tweets['nearest_latitude'] = c_x
        airline_tweets['nearest_longitude'] = c_y
        airline_tweets['nearest_city'] = c_n

        feature_df = airline_tweets[['nearest_city']]
        label_df = airline_tweets['airline_sentiment']

        # feature encoding
        features_to_csv = pd.get_dummies(
            feature_df, columns=["nearest_city"])

        # label encoding
        le = preprocessing.LabelEncoder()
        le.fit(label_df.values)
        labels = le.transform(label_df)
        features_to_csv['labels'] = labels
        features_to_csv.to_csv(self.output_file)


class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative,  neutral,  positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    feature_file = luigi.Parameter(default='features.csv')
    output_file = luigi.Parameter(default='model.pkl')
    numpy_x = luigi.Parameter(default='X_test')
    feature_names = luigi.Parameter(default='feature_names.csv')

    def requires(self):
        return TrainingDataTask(tweet_file=self.tweet_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        features = pd.read_csv(self.feature_file)
        labels = features['labels']
        labels = np.array(labels)
        features = features.drop(['labels'], axis=1)
        feature_names = features.columns.values
        feature_names = pd.Series(feature_names)
        feature_names.to_csv(self.feature_names)

        # split X and y into training and testing sets
        from sklearn.cross_validation import train_test_split
        X_train,  X_test,  y_train,  y_test = train_test_split(
            features, labels, test_size=0.40, random_state=0)
        X_train = np.array(X_train)

        # Avoid numpy array adding additional feature
        X_train = X_train[:, 1:]
        X_test = np.array(X_test)
        X_test = X_test[:, 1:]
        np.save(self.numpy_x,  X_test)

        # Fitting Logistic Model
        lr_model = LogisticRegression(class_weight="balanced", C=0.7)
        lr_model.fit(X_train,  y_train)
        lr_predict_class = lr_model.predict(X_test)
        print("Accuracy of the model: {0:.4f}".format(metrics.accuracy_score(
            y_test,  lr_predict_class)))

        # Save the scored model on to disk
        filename = self.output_file
        pickle.dump(lr_model,  open(filename, 'wb'))


class ScoreTask(luigi.Task):
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='scores.csv')
    pickle_file = luigi.Parameter(default='model.pkl')

    def requires(self):
        yield CleanDataTask(tweet_file=self.tweet_file)
        yield TrainingDataTask(tweet_file=self.tweet_file)
        yield TrainModelTask(tweet_file=self.tweet_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):

        # load the model from disk
        lr_model = pickle.load(open(self.pickle_file, 'rb'))
        feature_names = pd.read_csv(
            'feature_names.csv', skiprows=[0], header=None)
        feature_name_series = pd.Series(feature_names[1])
        X_test = np.load('X_test.npy')
        X_test = np.array(X_test)
        nb_predict_class = lr_model.predict(X_test)
        probabilities = lr_model.predict_proba(X_test)
        probabilities_df = pd.DataFrame(
            probabilities, columns=['negative', 'neutral', 'positive'])
        probabilities_df['cities'] = feature_name_series
        city_probability = probabilities_df[
            ['cities', 'negative', 'neutral', 'positive']]
        city_probability['model_predictions'] = nb_predict_class
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        city_probability['sentiment_prediction']\
            = city_probability['model_predictions'].map(sentiment_map)
        city_probability.to_csv('all_city_probability.csv')

        # Sorted list of non null cities by the predicted positive score
        predicted_non_null = city_probability[
            (city_probability['cities'].notnull())]
        sorted_predicted_positive = predicted_non_null.sort_values(
            ['positive'],  ascending=[False])
        sorted_predicted_positive.to_csv(self.output_file)

if __name__ == '__main__':
    luigi.run()
