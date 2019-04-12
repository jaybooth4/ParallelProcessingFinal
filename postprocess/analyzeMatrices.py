import pandas as pd
import argparse
import sys

BEERVECTORS = "testMatrix.csv"
BEERLOOKUP = "../data/beer/beerLookup.csv"

def parseArgs():
    parser = argparse.ArgumentParser(
        description='User beer preferences', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--beer', type=str, help="Preferred name of beer", nargs='*')
    parser.add_argument('--style', type=str, help="Preferred style of beer", nargs='*')
    args = parser.parse_args()
    if args.beer == None and args.style == None:
        print "Please enter one or multiple beers/styles"
        sys.exit(1)
    return args

def getBeerVectors(beerVectorFile):
    beerVectors = pd.read_csv(beerVectorFile)
    beerVectors['featureVector'] = beerVectors.iloc[:, 1:].values.tolist()
    beerVectors['beer_beerid'] = beerVectors.iloc[:, 0]
    beerVectors = beerVectors[['beer_beerid', 'featureVector']].set_index('beer_beerid')
    return beerVectors

def getBeerData(lookupFile, vectorFile):
    beerLookup = pd.read_csv(BEERLOOKUP).set_index('beer_beerid')
    beerVectors = getBeerVectors(vectorFile)
    beerData = beerLookup.join(beerVectors, how='inner')
    return beerData

def similarBeers(beerData, beers):
    return True

def similarStyles(beerData, style):
    return True

def main():
    args = parseArgs()
    beerData = getBeerData(BEERLOOKUP, BEERVECTORS)
    if args.beer:
        similarBeers(beerData, args.beer)
    else:
        similarStyles(beerData, args.style)

if __name__ == "__main__":
    main()
    