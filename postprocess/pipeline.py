from similarityGraph import similarityGraph
from matrixGrapher import MatrixGrapher

OUTPUT = '../results/'
BEERVECTORS = "../results/als/v.csv/"
BEERPROCESSEDDIR = "../data/beerProcessed/"
STYLES = [76, 79, 12]

# 14=APA,m 53=Euro Pale Lager, 76=light lager, 9=American double IPA
# 12=IPA, 103=Whitbier, 79=Milk sweet Stout

def main():
    # similarityGraph('style', BEERPROCESSEDDIR, OUTPUT)
    # similarityGraph('beer', BEERPROCESSEDDIR, OUTPUT)
    mg = MatrixGrapher(BEERVECTORS, BEERPROCESSEDDIR, OUTPUT, STYLES)
    mg.threeDPlot()
    # mg.graphPCA()
    # mg.graphTSNE(10)

if __name__ == "__main__":
    main()