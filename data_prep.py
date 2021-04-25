import json
import pandas as pd
import numpy as np
import fasttext

review_path = "data/goodreads_reviews_dedup.json"
book_path = "data/goodreads_books.json"

# read review data, removing entries with 0 likes
reviews = []
total_reviews = 0
with open(review_path) as f:
    for line in f:
        total_reviews += 1
        entry = json.loads(line)
        if entry['n_votes'] + entry['n_comments'] > 0:
            reviews.append(entry)

print("total_reviews: ", total_reviews)

# convert to DF, retaining only relevant columns
reviews = pd.DataFrame(reviews, columns=['user_id','book_id','rating','review_text','date_added','n_votes','n_comments'])
print("reviews with 1+ like/comment: ", len(reviews))

# calculate number of reviews per user
user_reviews = reviews.groupby("user_id")["book_id"].count().rename("user_reviews")

# combine votes and comments
reviews["review_likes"] = reviews["n_votes"] + reviews["n_comments"]
reviews = reviews.drop(["n_votes","n_comments"],axis=1)

# read book data, removing books with fewer than 10 reviews
books = []
total_books = 0
with open(book_path) as f:
    for line in f:
        total_books += 1
        entry = json.loads(line)
        try:
            if int(entry['text_reviews_count']) >= 10:
                books.append(entry)
        except:
            print("error for text reviews count value: ", entry['text_reviews_count'])

print("total_books: ", total_books)

# convert to DF, retaining only relevant columns
books = pd.DataFrame(books, columns=["book_id","text_reviews_count","ratings_count","average_rating"])
print("books with 10+ reviews: ", len(books))

# join reviews and books
dat = pd.merge(reviews,books,on="book_id")
print("join reviews/books: ", len(dat))

# calculate total review likes per book
book_review_likes = dat.groupby("book_id")["review_likes"].sum().rename("book_review_likes")

# remove books with fewer than 60 total likes on reviews
book_review_likes = book_review_likes[book_review_likes>=60]
dat = dat.merge(book_review_likes,on='book_id')
print("reviews on books with 60+ review likes: ", len(dat))

# calculate like share
dat["like_share"] = dat["review_likes"]/dat["book_review_likes"]

# create popularity binary variable
popular_thresh = 0.02
dat["popular"] = np.where(dat["like_share"]>popular_thresh,1,0)

print("pre english filter popularity count:")
print(dat.groupby("popular")["user_id"].count())

# filter to books in English
language_model = fasttext.load_model('lid.176.bin')
pred = language_model.predict(dat["review_text"].str.replace('\n','').to_list())
keep_ind = [i for i in range(len(pred[0])) if pred[0][i][0] == '__label__en' and pred[1][i][0] > .90]
dat = dat.iloc[keep_ind,].reset_index(drop=True)
print("english reviews: ", len(dat))

# create days since added column
review_dates = pd.to_datetime(pd.to_datetime(dat["date_added"],format='%a %b %d %H:%M:%S %z %Y',errors='coerce'),utc=True,errors='coerce')
dat["days_since_review"] = (max(review_dates) - review_dates).dt.days

# add user_reviews column
dat = dat.merge(user_reviews,on='user_id')

# rename and reorder columns
dat = dat.rename(columns={"rating": "user_rating", "text_reviews_count":"book_reviews", "average_rating":"avg_rating"})
dat = dat[["book_id", "user_reviews", "user_rating", "avg_rating", "ratings_count", "review_text", "days_since_review", "review_likes", "like_share", "popular"]]
print("final length: ", len(dat))

# save dataset
dat.to_csv("filtered_reviews.csv", index=False)