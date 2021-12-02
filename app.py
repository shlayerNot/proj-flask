from flask import Flask,request, render_template
###### Import fundamentals
import numpy as np
import pandas as pd
import re

#import warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import word_tokenize and stopwords from nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag


# Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

file = open("StopWords/flstopwords.txt", "r", encoding="utf8")
flstopwords = file.read().split("\n")
file.close()

#Naive Bayes Classifiers
nb = BernoulliNB()
mnb = MultinomialNB()
cnb = ComplementNB()
gnb = GaussianNB()

enstopwords = set(stopwords.words('english'))

def process_tweets(tweet):
    tweet = re.sub(r"won't", "will not", tweet)
    tweet = re.sub(r"can't", "can not", tweet)
    tweet = re.sub(r"n't", " not", tweet)
    tweet = re.sub(r"'ve", " have", tweet)
    tweet = re.sub(r"'ll", " will", tweet)
    tweet = re.sub(r"'re", " are", tweet)

    tweet = re.sub(r"'di", "hindi", tweet)
    tweet = re.sub(r"di", "hindi", tweet)

    # Remove links
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)

    # remove numbers
    tweet = re.sub(r'\d', '', tweet)

    # Remove mentions and hashtag
    tweet = re.sub(r'\@\w+|\#', '', tweet)

    # clean the words
    clean = word_tokenize(tweet)

    # Remove the English stop words
    clean = [token for token in clean if token not in enstopwords]

    # Remove the Filipino stop words
    clean = [token for token in clean if token not in flstopwords]

    # Remove non-alphabetic characters and keep the words contains three or more letters
    clean = [token for token in clean if token.isalpha() and len(token) > 2]

    clean = ' '.join(clean)

    return clean

def NormalizeWithPOS(text):
    # Lemmatization & Stemming according to POS tagging

    word_list = word_tokenize(text)
    rev = []
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for word, tag in pos_tag(word_list):
        if tag.startswith('J'):
            w = lemmatizer.lemmatize(word, pos='a')
        elif tag.startswith('V'):
            w = lemmatizer.lemmatize(word, pos='v')
        elif tag.startswith('N'):
            w = lemmatizer.lemmatize(word, pos='n')
        elif tag.startswith('R'):
            w = lemmatizer.lemmatize(word, pos='r')
        else:
            w = word
        w = stemmer.stem(w)
        rev.append(w)
    tweet = ' '.join(rev)

    return tweet

# Initialize Naive Bayes classifiers
def nbBernoulli(X_train, y_train):
    nb.fit(X_train, y_train)

    return nb

def nbComplement(X_train, y_train):
    cnb.fit(X_train, y_train)

    return cnb

def nbMultinomial(X_train, y_train):
    mnb.fit(X_train, y_train)

    return mnb

def nbGaussianNB(X_train, y_train):
    gnb.fit(X_train, y_train)

    return gnb

def predictT(cl, X_test):
    y_pred = cl.predict(X_test)

    return y_pred

# Evaluate the Model using confusion matrix
def confusionmt(cl, X_test, y_test, y_pred):
    # Predict the labels
    y_pred = cl.predict(X_test)

    # Print the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    #print("Confusion Matrix\n")
    #print(cm)

    # Print the Classification Report
    cr = classification_report(y_test, y_pred)
    #print("\n\nClassification Report\n")
    #print(cr)

    return cm, cr

def accuracy(cr):
    total = cr.sum().sum()
    accuracy = np.diag(cr).sum() / total
    accuracy = accuracy * 100
    accuracy = round(accuracy, 2)

    return accuracy

@app.route('/', methods=["GET", "POST"])
@app.route('/dashboard', methods=["GET", "POST"])
def dashboard():
    #load the dataset
    tweets = pd.read_csv("FinalDataset.csv")

    #clean tweets
    tweets["Processed"] = tweets["Tweets"].str.lower().apply(process_tweets)

    #text stemming
    tweets["Processed"] = tweets["Processed"].apply(NormalizeWithPOS)

    # Initialize a Tf-idf Vectorizer
    vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, stop_words=enstopwords and flstopwords)

    # Fit and transform the vectorizer corpus = [str (item) for item in corpus]
    tfidf_matrix = vectorizer.fit_transform(str(item) for item in tweets["Processed"])

    X = tfidf_matrix
    y = tweets["Label"]

    #split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.20)

    bnb = nbBernoulli(X_train, y_train)
    mnb = nbMultinomial(X_train, y_train)
    #gnb = nbGaussianNB(X_train, y_train)
    cnb = nbComplement(X_train, y_train)

    #gnbprediction = predictT(gnb, X_test)
    bnbprediction = predictT(bnb, X_test)
    cnbprediction = predictT(cnb, X_test)
    mnbprediction = predictT(mnb, X_test)

    #gnbconfusionmt = confusionmt(gnb, X_test, y_test, gnbprediction)
    bnbcm, cnbcr = confusionmt(bnb, X_test, y_test, bnbprediction)
    cnbcm, cnbcr = confusionmt(cnb, X_test, y_test, cnbprediction)
    mnbcm, mnbcr = confusionmt(mnb, X_test, y_test, mnbprediction)

    # Visualize the Label counts
    neutral = tweets.loc[tweets['Label'] == 'Neutral'].count()
    positive = tweets.loc[tweets['Label'] == 'Positive'].count()
    negative = tweets.loc[tweets['Label'] == 'Negative'].count()
    data = {'Label': 'Count', 'Neutral': neutral[0], 'Positive': positive[0], 'Negative': negative[0]}

    #total number of dataset
    total = neutral[0] + positive[0] + negative[0]


    #accuracy per algo
    Bnaive = accuracy(bnbcm)
    Mnaive = accuracy(mnbcm)
    Cnaive = accuracy(cnbcm)


    if request.method == "POST":

        tweet = request.form.get("tweet")
        clean = process_tweets(tweet)
        prediction = vectorizer.transform([tweet])

        prediction1 = nb.predict(prediction)
        prediction2 = mnb.predict(prediction)
        prediction3 = cnb.predict(prediction)
        #ps = mnb.predict_proba(prediction2[:1])
        #prediction4 = gnb.predict(prediction)
        return render_template('Dashboard.html',tweet= tweet, p1=prediction1[0], p2=prediction2[0], p3=prediction3[0], data=data,matrix=mnbcm, matrix1= mnbcr,
                               Bnaive = Bnaive,Mnaive = Mnaive, Cnaive = Cnaive, total = total)


    return render_template('Dashboard.html', data=data, Bnaive = Bnaive,Mnaive = Mnaive, Cnaive = Cnaive, total = total)


if __name__ == "__main__":
    app.run(debug=True)
