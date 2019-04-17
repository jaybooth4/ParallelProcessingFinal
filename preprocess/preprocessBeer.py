import pandas as pd
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import numpy as np

DATAFILE = "../data/raw/beer_reviews.csv"
OUTDATADIR = "../data/beer/"
BEERLOOKUPDIR = "../data/beerProcessed/"
NUMFOLDS = 5

def createBeerLookup(df, save=True):
    # Create a beer lookup table of ID to name
    beerData = df[['beer_beerid', 'beer_name', 'beer_style', 'review_overall']].copy()
    beerData['beer_style_id'] = beerData['beer_style'].astype('category').cat.codes
    beerNames = beerData.drop(columns=['review_overall']).drop_duplicates().set_index('beer_beerid')
    beerAvgRating= beerData.groupby('beer_beerid')[['beer_beerid', 'review_overall']].mean().set_index('beer_beerid')
    beerLookup = beerNames.join(beerAvgRating, how='inner')
    if save:
        beerLookup.to_csv(BEERLOOKUPDIR + "beerLookup.csv")
    return beerLookup

def createCollaborativeDataset(df, save=True):
    # (user, item, rating) format
    collaborativeDataset = df[['review_profilename', 'beer_beerid', 'review_overall']].copy()
    collaborativeDataset["review_profilename"] = collaborativeDataset["review_profilename"].astype('category').cat.codes
    collaborativeDataset = shuffle(collaborativeDataset)

    # Split datasets into folds
    splits = np.array_split(collaborativeDataset, NUMFOLDS)
    if save:
        for k, dataset in enumerate(splits):
            dataset.to_csv(OUTDATADIR + "fold" + str(k), index=False, header=False)
    return splits


def main():
    df = pd.read_csv(DATAFILE)
    
    # Highest NA is ~ 67000 for beer_abv
    # print df.isnull().sum(axis = 0)
    df.dropna(0, inplace=True)

    createBeerLookup(df)
    createCollaborativeDataset(df)

if __name__ == "__main__":
    main()