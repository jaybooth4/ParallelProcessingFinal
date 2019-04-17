import glob
import pandas as pd
import numpy as np

def readMultipartCSV(directory):
    files = set(glob.glob(directory + "*")) - set(glob.glob(directory + "*.crc")) - set(glob.glob(directory + "_SUCCESS"))
    return pd.concat(map(lambda f: pd.read_csv(f, header=None), files))

def getBeerIdsByStyles(styles, beerlookupDir):
    beerLookup = pd.read_csv(beerlookupDir + "beerLookup.csv")
    allIds = np.array([])
    mapIdsToStyle = {}
    for style in styles:
        listOfIds = beerLookup.loc[beerLookup['beer_style_id'] == style, 'beer_beerid'].values
        for beerId in listOfIds:
            mapIdsToStyle[beerId] = style
        allIds = np.concatenate([allIds, listOfIds])
    return allIds, mapIdsToStyle