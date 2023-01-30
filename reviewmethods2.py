import concurrent.futures
import json
import re
import string
import time
import urllib

import contractions
import fasttext
import nltk
import pandas as pd
import requests
import snowballstemmer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.download('vader_lexicon')
nltk.download('stopwords')
vader = SentimentIntensityAnalyzer()
model = fasttext.load_model('lid.176.ftz')
why_do_i_have_to_do_this0 = model.predict("I am so very fluent in english")
why_do_i_have_to_do_this = why_do_i_have_to_do_this0[0]
stop = stopwords.words('english')
data = {'recommendationid': [pd.NA], 'language': [pd.NA], 'review': [pd.NA], 'timestamp_created': [pd.NA],
        'timestamp_updated': [pd.NA], 'voted_up': [pd.NA], 'votes_up': [pd.NA], 'votes_funny': [pd.NA],
        'weighted_vote_score': [pd.NA], 'comment_count': [pd.NA], 'steam_purchase': [pd.NA],
        'received_for_free': [pd.NA], 'written_during_early_access': [pd.NA], 'hidden_in_steam_china': [pd.NA],
        'steam_china_location': [pd.NA], 'author.steamid': [pd.NA], 'author.num_games_owned': [pd.NA],
        'author.num_reviews': [pd.NA], 'author.playtime_forever': [pd.NA], 'author.playtime_last_two_weeks': [pd.NA],
        'author.playtime_at_review': [pd.NA], 'author.last_played': [pd.NA], 'cursor': [pd.NA],
        'query_summary.num_reviews': [pd.NA], 'appid': [pd.NA], 'review_type': [pd.NA]}
emptydf = pd.DataFrame(data)


def vader_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score >= 0:
        return 'positive'
    elif score < 0:
        return 'negative'


def textblob_sentiment(text):
    scores = vader.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score >= 0:
        return 'positive'
    elif compound_score < 0:
        return 'negative'


def clean0(text):  # Heaviest cleaning( removes any punctuation and numbers)
    text = text.lower()
    text = contractions.fix(text)
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = " ".join(text.split())  # remove extra whitespace and newline...
    text = text.translate(str.maketrans('', '', string.digits))
    return text


# cleaning unecessary text from the string von dem Juypter notebook
def clean(text):
    # cleanup
    text = re.sub('<.*?>+', ' ', text)  # removing HTML Tags
    text = re.sub('\n', ' ', text)  # removal of new line characters
    text = re.sub('\[', ' ', text)  # removal []
    text = re.sub('\]', ' ', text)  # removal ]
    text = re.sub(r'\s+', ' ', text)  # removal of multiple spaces
    # concatenate tokens
    return text

    # https://monkeylearn.com/blog/text-cleaning/


def clean2(text):
    text = text.lower()
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    text = ' '.join(text.split())
    return text


def clean1(text):  # Heavy cleaning( removes any punctuation)
    text = text.lower()
    text = contractions.fix(text)
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = " ".join(text.split())  # remove extra whitespace and newline...
    return text


def clean3(text):  # (removes only repeating punctuation)
    text = text.lower()
    text = contractions.fix(text)
    text = " ".join(text.split())  # remove extra whitespace and newline...
    text = re.sub(r'(\W)(?=\1)', '', text)
    return text


def remove_non_english(text):
    test = model.predict(text)
    if test[0] == why_do_i_have_to_do_this:
        return text
    text = ""
    return text


def remove_stopwords(text):
    text = ' '.join([word for word in text.split() if word not in stop])
    return text


def snowball_stemmer(text):
    # stemming
    stemmer = snowballstemmer.stemmer('english')
    stemmText = stemmer.stemWords(text.split())
    # concatenate tokens
    cleaned_text = " ".join([str(item) for item in stemmText])
    return cleaned_text


def get_top100games(reviews_per_game):
    x = requests.get('https://steamspy.com/api.php?request=top100in2weeks')
    binary = x.content
    df1 = pd.DataFrame(json.loads(binary)).loc[['appid', 'name', 'positive', 'negative'], :].T
    df1.reset_index(drop=True, inplace=True)
    df1['neg_multiplier'] = df1.negative / (df1.negative + df1.positive)
    df1['neg_multiplier'] = df1.neg_multiplier.astype(float).round(2)
    df1['negative_amount'] = df1.neg_multiplier * reviews_per_game
    df1['positive_amount'] = (1 - df1.neg_multiplier) * reviews_per_game
    df1['negative_amount'] = df1.negative_amount.astype(int)
    df1['positive_amount'] = df1.positive_amount.astype(int)
    df1.to_csv('output_top100games.csv', index=False)


def get_review(appid, amount, review_type, cursorInput):
    # print(f'Anfang: Appid: {appid} , amount: {amount}, review_type: {review_type}, cursorInput: {cursorInput}')
    r = requests.get(f"http://store.steampowered.com/appreviews/{appid}?filter=recent&language=english"
                     f"&review_type={review_type}&num_per_page={amount}&purchase_type=all&json=1&cursor={cursorInput}")
    print(f'Appid:{appid} mit der Menge:{amount} und review_type: {review_type} cursor: {cursorInput}')
    print(r.status_code)
    dfout = pd.json_normalize(r.json(), record_path=['reviews'], meta=['cursor', ['query_summary', 'num_reviews']])
    if dfout.empty:
        print(f'Error mit Appid: {appid} mit der Menge: {amount} und review_type: {review_type} cursor: {cursorInput}')
        emtpy = emptydf.copy()
        emtpy['appid'] = appid
        emtpy['review_type'] = review_type
        return emtpy
    dfout['appid'] = appid
    dfout['review_type'] = review_type
    # print(f'Vor if Appid: {appid} , amount: {amount}, review_type: {review_type}, cursorInput: {cursorInput}')

    return dfout


def get_reviews(appid, amount, review_type):
    if amount > 100:
        # print(f"Working on {review_type} reviews for: " + str(appid) + " amount= " + str(amount))
        df = get_review(appid, 100, review_type, '*')
        if df['cursor'] is not pd.NA:
            running_cursor = urllib.parse.quote(df.cursor[0].encode('utf8'))
        else:
            print(
                f'HEEEEEEEEEEEEEEEEEEEEEEEEEEEELP get_reviews Appid: {appid} mit der Menge: {amount} und review_type: {review_type} cursor: nix da')
            emtpy = emptydf.copy()
            emtpy['appid'] = appid
            emtpy['review_type'] = review_type
            return emtpy

        dfList = [df]
        amount -= 100
        while amount > 100:
            # print("while loop amount: " + str(amount) + " cursor: " + str(running_cursor))
            df = get_review(appid, 100, review_type, running_cursor)
            if df.cursor[0] is not pd.NA:
                running_cursor = urllib.parse.quote(df.cursor[0].encode('utf8'))
            else:
                print(
                    f'While HEEEEEEEEEEEEEEEEEEEEEEEEEEEELP get_reviews Appid: {appid} mit der Menge: {amount} und review_type: {review_type} cursor: nix da')
                emtpy = emptydf.copy()
                emtpy['appid'] = appid
                emtpy['review_type'] = review_type
                return emtpy
            dfList.append(df)
            amount -= 100
        try:
            if amount > 0:
                dfList.append(get_review(appid, amount, review_type, running_cursor))
        except:
            print('error')
        out = dfList[0]
        for other_df in dfList[1:]:
            # print('for loop')
            out = pd.concat([out, other_df])
        out = out.reset_index()
        out = out.drop('index', axis=1)
        return out
    return get_review(amount, appid, review_type, '*')


def reviews_to_dataframe(appid_list, negative_list, positive_list):
    dflist = []
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_reviews = {executor.submit(get_reviews, ids, number, 'negative'):
                              (ids, number) for ids, number in zip(appid_list, negative_list)}
        for future in concurrent.futures.as_completed(future_reviews):
            ids, number = future_reviews[future]
            try:
                df = future.result()
                dflist.append(df)
            except:
                print(f'Alaaaaaaaaaaaaaaaaaarm to_df type: negative Appid: {ids} amount: {number}')
    end = time.time()
    print('Time taken for negative reviews: ' + str(end - start))
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_reviews = {executor.submit(get_reviews, ids, number, 'positive'):
                              (ids, number) for ids, number in zip(appid_list, positive_list)}
        for future in concurrent.futures.as_completed(future_reviews):
            ids, number = future_reviews[future]
            try:
                df = future.result()
                dflist.append(df)
            except:
                print(f'Alaaaaaaaaaaaaaaaaaarm to_df type: positive Appid: {ids} amount: {number}')
    end = time.time()
    print('Time taken for positive reviews: ' + str(end - start))
    start = time.time()
    out = dflist[0]
    for other_df in dflist[1:]:
        out = pd.concat([out, other_df])
    out = out.reset_index()
    out.drop(['index', 'cursor'], axis=1, inplace=True)
    end = time.time()
    print('Time taken for concatenation: ' + str(end - start))
    return out
