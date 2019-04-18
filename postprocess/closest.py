import pandas as pd
import argparse
import sys
from scipy.spatial.distance import euclidean
import numpy as np
from util import readMultipartCSV

BEERVECTORS = "../results/als/v.csv/"
BEERLOOKUP = "../data/beerProcessed/beerLookup.csv"
OUTPUT = "../data/beerProcessed/"

def getBeerVectorData(beerVectorFile):
    ''' Get beer vectors from output of training run '''
    beerVectors = readMultipartCSV(BEERVECTORS)
    beerVectors['featureVector'] = beerVectors.iloc[:, 1:].values.tolist()
    beerVectors['beer_beerid'] = beerVectors.iloc[:, 0]
    beerVectors = beerVectors[['beer_beerid', 'featureVector']].set_index('beer_beerid')
    beerVectors['featureVector'] = beerVectors['featureVector'].apply(lambda vector: np.array(vector))
    return beerVectors

def getBeerData(lookupFile, vectorFile):
    ''' Get beer dataframe '''
    beerLookup = pd.read_csv(BEERLOOKUP).set_index('beer_beerid')
    beerVectors = getBeerVectorData(vectorFile)
    beerData = beerLookup.join(beerVectors, how='inner')
    return beerData

def closestBeersToVector(beerData, vector, save=True):
    ''' Get closest beers to a given vector '''
    beerDistances = beerData[['featureVector']].copy()
    beerDistances['distance'] = list(map(lambda beer: np.linalg.norm(beer - vector), beerDistances['featureVector']))
    beerDistances.drop(columns='featureVector', inplace=True)
    beerDistances = beerDistances.join(beerData, how='inner')[['beer_name', 'beer_style', 'distance']].sort_values(by=['distance'])
    beerDistances = beerDistances[beerDistances['distance'] != 0]
    if save:
        beerDistances.to_csv(OUTPUT + "beersDistances.csv")
    return beerDistances

def getBeerVector(beerData, beers):
    ''' Get vector representation for one or more beers '''
    # For some reason .values on the series is returning a list of list of lists
    specifiedBeers = [beerVector[0] for beerVector in beerData.loc[beers, ['featureVector']].values]
    namesVals = beerData.loc[beers, 'beer_name'].values
    names = ", ".join(namesVals)
    return np.mean(specifiedBeers, axis=0), names

def closestStylesToVector(beerData, vector, save=True):
    ''' Get closest styles to a given vector '''
    styleVectors = beerData.groupby("beer_style_id")["featureVector"]\
        .apply(list).apply(lambda listOfVectors: np.mean(listOfVectors, axis=0))
    styleDistances = styleVectors.apply(lambda style: np.linalg.norm(style - vector))
    styleDistances = pd.DataFrame({'beer_style_id':styleDistances.index, 'distance':styleDistances.values}).set_index('beer_style_id')
    styleDistances = styleDistances.join(beerData[['beer_style_id', 'beer_style']].drop_duplicates().set_index("beer_style_id"), how='inner')
    styleDistances.sort_values(by=['distance'], inplace=True)
    styleDistances = styleDistances[styleDistances['distance'] != 0]
    if save:
        styleDistances.to_csv(OUTPUT + "stylesDistances.csv")
    return styleDistances

def getStyleVector(beerData, styles):
    ''' Get vector representation for one or more styles '''
    styleVectors = beerData.groupby("beer_style_id")["featureVector"]\
            .apply(list).apply(lambda listOfVectors: np.mean(listOfVectors, axis=0))
    namesVals = beerData.loc[beerData['beer_style_id'].isin(styles)][:, 'beer_style'].unique().values
    names = ', '.join(namesVals)
    return styleVectors.loc[styles].mean(), names

def topBeers(beerData, styles, save=True):
    ''' Returns a sorted list of beers '''
    topBeersByStyle = beerData.loc[beerData['beer_style_id'].isin(styles)].groupby('beer_beerid')[['beer_style_id', 'review_overall']].mean()\
        .sort_values(by=['review_overall'], ascending=False)
    topBeersByStyle = topBeersByStyle.join(beerData[['beer_name']].drop_duplicates())
    if save:
        topBeersByStyle.to_csv(OUTPUT + "topBeersByStyle.csv")
    return topBeersByStyle