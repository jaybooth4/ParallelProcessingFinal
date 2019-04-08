# -*- coding: utf-8 -*-
import numpy as np
from time import time
import sys
import argparse
from pyspark import SparkContext
from operator import add
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession

def parseArgs():
    """ 
        Arg Parsing.
    """
    parser = argparse.ArgumentParser(
        description='Alternating least squares.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', help='Directory containing folds. The folds should be named fold0, fold1, ..., foldK.')
    parser.add_argument('folds', type=int, help='Number of folds')
    parser.add_argument('--reg', default=1.0, type=float, help="Regularization parameter")
    parser.add_argument('--d', default=10, type=int, help="Number of latent features")
    parser.add_argument('--outputfile', help='Output file to save the model')
    parser.add_argument('--iter', default=20, type=int, help='Number of iterations to use during training')
    parser.add_argument('--N', default=40, type=int, help='Parallelization Level')
    parser.add_argument('--master',default="local[4]",help="Spark Master")

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)

    return parser.parse_args()

def readRatings(fileName, sparkContext, sess):
    """ 
        Read in ratings from a given file. 
        Assumes the file is a csv in the format (user, item, rating)
    """
    ratingsRDD = sparkContext.textFile(fileName).map(lambda x: tuple(x.split(',')))\
        .map(lambda (user, item, rating): Row(userId=int(user), itemId=int(item), rating=float(rating)))
    return sess.createDataFrame(ratingsRDD)

def readFolds(directory, numFolds, sc, sess):
    """
        Reads folds of data from a directory.
        Assumes files are formatted with "fold[i]" where i is the fold number
    """
    folds = {}
    for k in range(numFolds):
        folds[k] = readRatings(directory+"/fold"+str(k), sc, sess)
    return folds

def createTrainTestData(folds, k, N):
    """
        Generates test and train data with the given fold object and value for k.
        k represents the fold to use for testing, while all other folds will be used for training.
    """
    train_folds = [folds[j] for j in folds if j is not k]
    train = train_folds[0]
    for fold in train_folds[1:]:
        train = train.union(fold)
    train.repartition(N).cache()
    test = folds[k].repartition(N).cache()
    return train, test

def main():
    args = parseArgs()
    sc = SparkContext(args.master, appName='Alternating least squares')
    sess = SparkSession(sc)

    if not args.verbose:
        sc.setLogLevel("ERROR")
    
    sc.setCheckpointDir('checkpoint/')

    folds = readFolds(args.data, args.folds, sc, sess)
    cross_val_rmses = []
    for k in range(len(folds)):
        train, test = createTrainTestData(folds, k, args.N)
        print"Initiating fold %d with %d train samples and %d test samples" % (k, train.count(), train.count())

        start = time()
        als = ALS(maxIter=args.iter, regParam=args.reg, userCol="userId", itemCol="itemId", ratingCol="rating", coldStartStrategy='drop')
        model = als.fit(train)
        predictions = model.transform(test)
        predictions.show()
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        testRMSE = evaluator.evaluate(predictions)

        now = time()-start
        print "Fold: %d\tTime: %f\tTestRMSE: %f" % (k, now, testRMSE)
        
        cross_val_rmses.append(testRMSE)
        train.unpersist()
        test.unpersist()

    print "%d-fold cross validation error is: %f " % (args.folds, np.mean(cross_val_rmses))

if __name__ == "__main__":
    main()