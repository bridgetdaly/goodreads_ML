# Text-Based Prediction of Book Review Popularity

Final project for Machine Learning (STATS229) Stanford Spring 2021 with goal of classifying book reviews as popular or unpopular based on share of likes and comments. Dataset comes from the [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home), originally scraped from [Goodreads](https://www.goodreads.com/).

#### Repo Contents

* data_prep.py: filter raw datasets
* features.py: feature engineering
* logistic_ab.py: logistic regression full sample and under sample on feature subsets a and b
* logistic_cd.py: logistic regression under sample on feature subsets c and d
* gb.py: gradient boosting (XGBoost) full sample on feature subsets a and b
* gb_under.py: gradient boosting (XGBoost) under sample on feature subsets a, b, c, and d
* neuralnet.py: neural net full sample on feature subsets a and b
* neuralnet_undersample.py: neural net under sample on feature subsets c and d
* save_undersample.py: save under sample train and test sets for smaller upload to GCP for next two scripts
* neuralnet_bow.py: neural net under sample on feature subset c
* neuralnet_tfidf.py: neural net under sample on feature subset d
* explore_predictions.ipynb: explore predictions from logistic subset b and random forest subset d
