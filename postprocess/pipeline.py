import argparse
import sys

from closest import getBeerData, getBeerVector, getStyleVector, closestBeersToVector, closestStylesToVector
from similarityGraph import similarityGraph
from matrixGrapher import MatrixGrapher

OUTPUT = '../results/'
BEERVECTORS = "../results/als/v2.csv/"
BEERPROCESSEDDIR = "../data/beerProcessed/"
BEERLOOKUP = '../data/beer/beerLookup.csv'
# STYLES = [76, 12, 13]
# STYLES = [76, 12, 77]
# STYLES = [76, 18, 77]
STYLES = [24, 4, 13]

# 50 stout
# 14=APA,m 53=Euro Pale Lager, 76=light lager, 
# 9=American double IPA, 18=AMerican stout
# 13=Malt, 4=barleywine, 24=Beglian Pale ale
# 12=American IPA, 103=Whitbier, 79=Milk sweet Stout

def parseArgs():
    ''' Parse input beer/style preferences '''
    parser = argparse.ArgumentParser(
        description='User beer preferences', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--beer', type=int, help="Preferred name of beer", nargs='*')
    parser.add_argument('--style', type=int, help="Preferred style of beer", nargs='*')
    args = parser.parse_args()
    return args

def main():
    args = parseArgs()
    if args.beer or args.style:
        beerData = getBeerData(BEERLOOKUP, BEERVECTORS)
        if args.beer:
            vector, names = getBeerVector(beerData, args.beer)
        else:
            vector, names = getStyleVector(beerData, args.style)
        closestBeers = closestBeersToVector(beerData, vector)
        closestStyles = closestStylesToVector(beerData, vector)

    # similarityGraph('Styles', names, OUTPUT, closestStyles)
    # similarityGraph('Beers', names, OUTPUT, closestBeers)
    mg = MatrixGrapher(BEERVECTORS, BEERLOOKUP, OUTPUT, STYLES)
    # mg.graphPCA()
    mg.graph3d()
    # mg.graphTSNE(25)

if __name__ == "__main__":
    main()