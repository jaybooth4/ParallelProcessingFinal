import pandas as pd

INPUTFILE = "testMatrix.csv"
BEERLOOKUP = "../data/beer/beerLookup.csv"

def getBeerVectors():
    beerVectors = pd.read_csv(INPUTFILE)
    beerVectors['featureVector'] = beerVectors.iloc[:, 1:].values.tolist()
    beerVectors['beer_beerid'] = beerVectors.iloc[:, 0]
    beerVectors = beerVectors[['beer_beerid', 'featureVector']].set_index('beer_beerid')
    return beerVectors

def main():
    beerLookup = pd.read_csv(BEERLOOKUP).set_index('beer_beerid')
    beerVectors = getBeerVectors()
    
    beerData = beerLookup.join(beerVectors, how='inner')
    print beerData

if __name__ == "__main__":
    main()