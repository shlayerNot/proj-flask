from flask import Flask,request, render_template
#Import fundamentals
import numpy as np
import pandas as pd
import re
import pickle

# Import word_tokenize and stopwords from nltk
import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag

#app = Flask(__name__)
app=Flask(__name__,template_folder='template')

file = open("StopWords/flstopwords.txt", "r", encoding="utf8")
flstopwords = file.read().split("\n")
file.close()

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
    #stemmer = PorterStemmer()
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
        #w = stemmer.stem(w)
        rev.append(w)
    tweet = ' '.join(rev)
    return tweet


@app.route('/', methods=["GET", "POST"])
@app.route('/dashboard', methods=["GET", "POST"])
def dashboard():
    #load the dataset
    tweets = pd.read_csv("Dataset.csv")

    # Initialize a Tf-idf Vectorizer
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

    # Visualize the Label counts
    neutral = tweets.loc[tweets['Label'] == 'Neutral'].count()
    positive = tweets.loc[tweets['Label'] == 'Positive'].count()
    negative = tweets.loc[tweets['Label'] == 'Negative'].count()
    data = {'Label': 'Count', 'Neutral': neutral[0], 'Positive': positive[0], 'Negative': negative[0]}

    #total number of dataset
    total = neutral[0] + positive[0] + negative[0]

    #accuracy = pickle.load(open('acc.pkl', 'rb'))

    if request.method == "POST":

        tweet = request.form.get("tweet")
        clean = process_tweets(str.lower(tweet))
        clean = NormalizeWithPOS(clean)
        prediction = vectorizer.transform([clean])

        bnb = pickle.load(open('BNB_model.pkl', 'rb'))
        prediction1 = bnb.predict(prediction)
        return render_template('Dashboard.html',tweet= tweet, p1=prediction1[0],data=data, total = total)

    return render_template('Dashboard.html', data=data,total = total)

if __name__ == "__main__":
    app.run(debug=True)