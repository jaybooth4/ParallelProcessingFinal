from similarityGraph import similarityGraph
from matrixGrapher import MatrixGrapher

def main():
    # similarityGraph('style')
    mg = MatrixGrapher()
    # mg.graphPCA()
    mg.graphTSNE()

if __name__ == "__main__":
    main()