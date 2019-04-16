import pandas as pd
import argparse
import sys
from scipy.spatial.distance import euclidean
import numpy as np

BEERVECTORS = "testMatrix.csv"
BEERLOOKUP = "../data/beer/beerLookup.csv"
OUTPUT = "../output/"

def parseArgs():
    ''' Parse input beer/style preferences '''
    parser = argparse.ArgumentParser(
        description='User beer preferences', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--beer', type=int, help="Preferred name of beer", nargs='*')
    parser.add_argument('--style', type=int, help="Preferred style of beer", nargs='*')
    args = parser.parse_args()
    if args.beer == None and args.style == None:
        print "Please enter one or multiple beers/styles"
        sys.exit(1)
    return args

def getBeerVectors(beerVectorFile):
    ''' Get beer vectors from output of training run '''
    beerVectors = pd.read_csv(beerVectorFile)
    beerVectors['featureVector'] = beerVectors.iloc[:, 1:].values.tolist()
    beerVectors['beer_beerid'] = beerVectors.iloc[:, 0]
    beerVectors = beerVectors[['beer_beerid', 'featureVector']].set_index('beer_beerid')
    beerVectors['featureVector'] = beerVectors['featureVector'].apply(lambda vector: np.array(vector))
    return beerVectors

def getBeerData(lookupFile, vectorFile):
    ''' Get beer dataframe '''
    beerLookup = pd.read_csv(BEERLOOKUP).set_index('beer_beerid')
    beerVectors = getBeerVectors(vectorFile)
    beerData = beerLookup.join(beerVectors, how='inner')
    return beerData

def getClosestBeers(beerData, beerVector):
    ''' Get closest beers to a given vector '''
    beerDistances = beerData[['featureVector']].copy()
    beerDistances['distance'] = list(map(lambda beer: np.linalg.norm(beer - beerVector), beerDistances['featureVector']))
    beerDistances.drop(columns='featureVector', inplace=True)
    beerDistances = beerDistances.join(beerData, how='inner')[['beer_name', 'beer_style', 'distance']].sort_values(by=['distance'])
    return beerDistances

def similarBeers(beerData, beers, save=True):
    ''' Get similar Beers to a given set of beers '''
    # For some reason .values on the series is returning a list of list of lists
    specifiedBeers = [beerVector[0] for beerVector in beerData.loc[beers, ['featureVector']].values]
    beerAvg = np.mean(specifiedBeers, axis=0)
    closestBeers = getClosestBeers(beerData, beerAvg)
    if save:
        closestBeers.to_csv(OUTPUT + "beerDistances.csv")
    return closestBeers

def topBeers(beerData, styles, save=True):
    ''' Returns a sorted list of beers '''
    topBeersByStyle = beerData.loc[beerData['beer_style_id'].isin(styles)].groupby('beer_beerid')[['beer_style_id', 'review_overall']].mean()\
        .sort_values(by=['review_overall'], ascending=False)
    topBeersByStyle = topBeersByStyle.join(beerData[['beer_name']].drop_duplicates())
    if save:
        topBeersByStyle.to_csv(OUTPUT + "topBeersByStyle.csv")
    return topBeersByStyle

def similarStyles(beerData, styles, save=True):
    ''' Returns most similar styles, and top beers per style '''
    styleVectors = beerData.groupby("beer_style_id")["featureVector"]\
            .apply(list).apply(lambda listOfVectors: np.mean(listOfVectors, axis=0))
    selectedStyleVectors = styleVectors.loc[styles].mean()
    styleDistances = styleVectors.apply(lambda style: np.linalg.norm(style - selectedStyleVectors))
    styleDistances = pd.DataFrame({'beer_style_id':styleDistances.index, 'distance':styleDistances.values}).set_index('beer_style_id')
    styleDistances = styleDistances.join(beerData[['beer_style_id', 'beer_style']].drop_duplicates().set_index("beer_style_id"), how='inner')
    styleDistances.sort_values(by=['distance'], inplace=True)
    if save:
        styleDistances.to_csv(OUTPUT + "styleDistances.csv")
    return styleDistances

def main():
    args = parseArgs()
    beerData = getBeerData(BEERLOOKUP, BEERVECTORS)
    if args.beer:
        print similarBeers(beerData, args.beer)
    else:
        print similarStyles(beerData, args.style)
        print topBeers(beerData, args.style)

if __name__ == "__main__":
    main()