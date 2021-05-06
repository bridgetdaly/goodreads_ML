import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download(['punkt','averaged_perceptron_tagger','vader_lexicon','stopwords','wordnet'])
pd.options.mode.chained_assignment = None

# read review data
dat = pd.read_csv("data/filtered_reviews.csv")
dat = dat.drop(columns=['book_id','ratings_count','review_likes','like_share'])

# difference between user rating and average book rating
dat["rating_diff"] = dat["user_rating"]-dat["avg_rating"]
dat = dat.drop(columns=['user_rating','avg_rating'])

# flag if review contains a quotation
dat["quote"] = dat["review_text"].str.contains("\"")

# tokenize for review length (num words), avg sentence length, avg word length
dat["tokenized_sents"] = dat["review_text"].apply(nltk.tokenize.sent_tokenize)
dat["num_sentences"] = dat["tokenized_sents"].apply(len)
dat["tokenized_words"] = dat["review_text"].apply(lambda review: [word.lower() for word in nltk.tokenize.word_tokenize(review) if word.isalpha()])
dat["num_words"] = dat["tokenized_words"].apply(len)
dat["avg_sent_len"] = dat["num_words"]/dat["num_sentences"]
dat["num_letters"] = dat["tokenized_words"].apply(lambda review: len([letter for word in review for letter in word]))
dat["avg_word_len"] = dat["num_letters"]/dat["num_words"]
dat = dat.drop(columns=['review_text','num_sentences','num_letters'])

# part of speech tagging
dat["pos_tags"] = dat["tokenized_words"].apply(nltk.pos_tag)

def count_pos(pos_tags, pos):
    counts = 0
    for word in pos_tags:
        if word[1][0] in pos:
            counts += 1
    return(counts)

dat["verbs"] = dat["pos_tags"].apply(count_pos,pos=["V"])
dat["pct_verbs"] = dat["verbs"]/dat["num_words"]
dat["nouns"] = dat["pos_tags"].apply(count_pos,pos=["N"])
dat["pct_nouns"] = dat["nouns"]/dat["num_words"]
dat["adj"] = dat["pos_tags"].apply(count_pos,pos=["J","R"])
dat["pct_adj"] = dat["adj"]/dat["num_words"]
dat = dat.drop(columns=['pos_tags','verbs','nouns','adj'])

# sentiment analysis
sid = SentimentIntensityAnalyzer()

def review_sentiment(review_sents):
    comptot = 0
    for sentence in review_sents:
        scores = sid.polarity_scores(sentence)
        comptot += scores['compound']
    return(comptot/len(review_sents))

dat["sentiment"] = dat["tokenized_sents"].apply(review_sentiment)
dat = dat.drop(columns=['tokenized_sents'])

# further text processing
# remove stop words
stopwords = nltk.corpus.stopwords.words('english')
dat["tokenized_words"] = dat["tokenized_words"].apply(lambda review: [word for word in review if word not in stopwords])
# lemmatization
wnl = nltk.stem.wordnet.WordNetLemmatizer()
dat["tokenized_words"] = dat["tokenized_words"].apply(lambda review: [wnl.lemmatize(word) for word in review])

# save dataset
dat.to_csv("tokenized_reviews.csv", index=False)