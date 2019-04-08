import pandas as pd
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import numpy as np

DATAFILE = "../data/raw/beer_reviews.csv"
OUTDATADIR = "../data/beer/"
NUMFOLDS = 5

def main():
    df = pd.read_csv(DATAFILE)
    # Highest NA is ~ 67000 for beer_abv
    # print df.isnull().sum(axis = 0)

    df.dropna(0, inplace=True)

    # Group by beer styles?
    # print df.groupby('beer_style', as_index=False)['beer_beerid'].count()

    # Create a beer lookup table of ID to name
    beerLookup = df.loc[:, ['beer_beerid', 'beer_name']].copy()
    beerLookup.drop_duplicates(inplace=True)
    # print beerLookup

    # Drop non-numeric values (assumes we don't want to use brewery info)
    df.drop(['brewery_id', 'brewery_name', 'review_time', 'beer_name'], axis=1, inplace=True)
    # print df.describe()

    userReviews = df.groupby(['review_profilename', 'beer_style'], as_index=False)[['review_overall', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'beer_abv']].mean()
    beerReviews = df.groupby(['beer_beerid', 'beer_style'], as_index=False)[['review_overall', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'beer_abv']].mean()
    # print(userReviews)
    # print(beerReviews)

    # (user, item, rating) format
    collaborativeDataset = df.loc[:, ['review_profilename', 'beer_beerid', 'review_overall']].copy()
    collaborativeDataset["review_profilename"] = collaborativeDataset["review_profilename"].astype('category').cat.codes
    collaborativeDataset = shuffle(collaborativeDataset)
    # print collaborativeDataset

    splits = np.array_split(collaborativeDataset, NUMFOLDS)
    for k, dataset in enumerate(splits):
        dataset.to_csv(OUTDATADIR + "fold" + str(k), index=False, header=False)

if __name__ == "__main__":
    main()