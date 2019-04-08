# -*- coding: utf-8 -*-
import numpy as np
from time import time
import sys
import argparse
from pyspark import SparkContext
from operator import add
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

def parseArgs():
    parser = argparse.ArgumentParser(
        description='Alternating least squares.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', help='Directory containing folds. The folds should be named fold0, fold1, ..., foldK.')
    parser.add_argument('folds', type=int, help='Number of folds')
    parser.add_argument('--reg', default=1.0, type=float, help="Regularization parameter")
    parser.add_argument('--d', default=10, type=int, help="Number of latent features")
    parser.add_argument('--outputfile', help='Output file to save the model')
    parser.add_argument('--maxiter', default=20, type=int, help='Maximum number of iterations')
    parser.add_argument('--N', default=40, type=int, help='Parallelization Level')
    parser.add_argument('--seed', default=1234567, type=int, help='Seed used in random number generator')
    parser.add_argument('--master',default="local[4]",help="Spark Master")

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)

    return parser.parse_args()

def readRatings(fileName, sparkContext):
    return sparkContext.textFile(fileName).map(lambda x: tuple(x.split(','))).map(lambda (i, j, rij): Rating(int(i), int(j), float(rij)))

def readFolds(directory, numFolds):
    folds = {}
    for k in range(numFolds):
        folds[k] = readRatings(directory+"/fold"+str(k), sc)
    return folds

def createTrainTestData(folds, k):
    train_folds = [folds[j] for j in folds if j is not k]
    train = train_folds[0]
    for fold in train_folds[1:]:
        train = train.union(fold)
    train.repartition(args.N).cache()
    test = folds[k].repartition(args.N).cache()
    return train, test


def testModel(model, testFold):
    featuresOnly = testFold.map(lambda rating: (rating[0], rating[1]))
    predictions = model.predictAll(featuresOnly).map(lambda rating: ((rating[0], rating[1]), rating[2]))
    ratings = testFold.map(lambda rating: ((rating[0], rating[1]), rating[2]))
    ratingsAndPredictions = ratings.join(predictions)
    MSE = ratingsAndPredictions.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    return MSE

if __name__ == "__main__":

    args = parseArgs()
    sc = SparkContext(args.master, appName='Alternating least squares')
    if not args.verbose:
        sc.setLogLevel("ERROR")
    
    sc.setCheckpointDir('checkpoint/')

    folds = readFolds(args.data, args.folds)
    
    cross_val_mses = []
    for k in range(len(folds)):
        train, test = createTrainTestData(folds, k)
        print"Initiating fold %d with %d train samples and %d test samples" % (k, train.count(), train.count())

        start = time()
        model = ALS.train(train, args.d, args.maxiter, args.reg)
        testMSE = testModel(model, test)
        now = time()-start
        print "Fold: %f\tTime: %f\tTestMSE: %f" % (k, now, testMSE)
        
        cross_val_mses.append(testMSE)
        train.unpersist()
        test.unpersist()

    print "%d-fold cross validation error is: %f " % (args.folds, np.mean(cross_val_mses))