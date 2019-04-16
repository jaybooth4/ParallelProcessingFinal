# -*- coding: utf-8 -*-
import numpy as np
from time import time
import sys,argparse
from pyspark import SparkContext
from pyspark.sql import SparkSession
from operator import add
from pyspark.mllib.random import RandomRDDs
import pickle
import pandas

def swap((x,y)):
    return (y,x)

def predict(u,v):
    return np.dot(u, v)

def pred_diff(r,u,v):
    return predict(u, v) - r    

def gradient_u(delta,u,v):
    return 2 * delta * v

def gradient_v(delta,u,v):
    return 2 * delta * u

def readRatings(file,sparkContext):
    return sparkContext.textFile(file).map(lambda x: tuple(x.split(','))).map(lambda (i,j,rij):(int(i),int(j),float(rij)))

def generateUserProfiles(R,d,seed,sparkContext,N):
    # exctract user ids
    U = R.map(lambda (i,j,rij):i).distinct(numPartitions = N)
    numUsers = U.count()
    randRDD = RandomRDDs.normalVectorRDD(sparkContext, numUsers, d,numPartitions=N, seed=seed)
    U = U.zipWithIndex().map(swap)
    randRDD = randRDD.zipWithIndex().map(swap)
    return U.join(randRDD,numPartitions = N).values()

def generateItemProfiles(R,d,seed,sparkContext,N):
    # extract item ids
    V = R.map(lambda (i,j,rij):j).distinct(numPartitions = N)
    numItems = V.count()
    randRDD = RandomRDDs.normalVectorRDD(sparkContext, numItems, d,numPartitions=N, seed=seed)
    V = V.zipWithIndex().map(swap)
    randRDD = randRDD.zipWithIndex().map(swap)
    return V.join(randRDD,numPartitions = N).values()

def joinAndPredictAll(R,U,V,N):
    return (R.map(lambda (i, j, rij): (i, (j, rij)))
             .join(U, numPartitions = N)
             .map(lambda (i, ((j, rij), ui)): (j, (i, ui, rij)))
             .join(V, numPartitions = N)
             .map(lambda (j, ((i, ui, rij), vj)): (i, j, pred_diff(rij, ui, vj), ui, vj)))


def SE(joinedRDD):
    return joinedRDD.map(lambda (i, j, dij, ui, vj): dij**2).sum()

def normSqRDD(profileRDD,param):
    return param * profileRDD.map(lambda (i, ui): np.linalg.norm(ui)**2).sum()


def adaptU(joinedRDD,gamma,lam,N):
    return (joinedRDD.map(lambda (i, _, dij, ui, vj): (i, (ui, gradient_u(dij, ui, vj))))
                     .reduceByKey(lambda (ui, grad1), (_, grad2): (ui, grad1 + grad2), numPartitions = N)
                     .map(lambda (i, (ui, grad)): (i, ui - gamma * (grad + 2 * lam * ui))))


def adaptV(joinedRDD,gamma,mu,N):
    return (joinedRDD.map(lambda (_, j, dij, ui, vj): (j, (vj, gradient_v(dij, ui, vj))))
                     .reduceByKey(lambda (vj, grad1), (_, grad2): (vj, grad1 + grad2), numPartitions = N)
                     .map(lambda (j, (vj, grad)): (j, vj - gamma * (grad + 2 * mu * vj))))

def get_args(data, 
            folds, 
            gain = 0.001, 
            power = 0.2, 
            epsilon = 1e-99, 
            lam = 1.0, 
            mu = 1.0, 
            d = 10, 
            outputfile = None,
            maxiter = 20, 
            N = 40, 
            seed = 1234567, 
            output = None, 
            verbose=False):
    '''
    Parallele Matrix Factorization.

    positional arguments:
      data                  Directory containing folds. The folds should be named
                            fold0, fold1, ..., foldK.
      folds                 Number of folds

    optional arguments:
      -h, --help            show this help message and exit
      --gain GAIN           Gain (default: 0.001)
      --power POWER         Gain Exponent (default: 0.2)
      --epsilon EPSILON     Desired objective accuracy (default: 1e-99)
      --lam LAM             Regularization parameter for user features (default:
                            1.0)
      --mu MU               Regularization parameter for item features (default:
                            1.0)
      --d D                 Number of latent features (default: 10)
      --outputfile OUTPUTFILE
                            Output file (default: None)
      --maxiter MAXITER     Maximum number of iterations (default: 20)
      --N N                 Parallelization Level (default: 40)
      --seed SEED           Seed used in random number generator (default:
                            1234567)
      --output OUTPUT       If not None, cross validation is skipped, and U,V are
                            trained over entire dataset and store it in files
                            output_U and output_V (default: None)
      --verbose
      --silent
    '''
    import sys

    sys.argv = ['main',
                str(data),
                str(folds),
                '--gain', str(gain),
                '--power', str(power),
                '--epsilon', str(epsilon),
                '--lam', str(lam),
                '--mu', str(mu),
                '--d', str(d),
                '--maxiter', str(maxiter),
                '--N', str(N),
                '--seed', str(seed)]
    
    if outputfile is not None:
        sys.argv.append('--outputfile', str(outputfile))
        
    if output is not None:
        sys.argv.append('--output', str(output))
        
    if verbose:
        sys.argv.append('--verbose')
    else:
        sys.argv.append('--silent')
        
    parser = argparse.ArgumentParser(description = 'Parallele Matrix Factorization.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data',help = 'Directory containing folds. The folds should be named fold0, fold1, ..., foldK.')
    parser.add_argument('folds',type = int,help = 'Number of folds')
    parser.add_argument('--gain',default=0.001,type=float,help ="Gain")
    parser.add_argument('--power',default=0.2,type=float,help ="Gain Exponent")
    parser.add_argument('--epsilon',default=1.e-99,type=float,help ="Desired objective accuracy")
    parser.add_argument('--lam',default=1.0,type=float,help ="Regularization parameter for user features")
    parser.add_argument('--mu',default=1.0,type=float,help ="Regularization parameter for item features")
    parser.add_argument('--d',default=10,type=int,help ="Number of latent features")
    parser.add_argument('--outputfile',help = 'Output file')
    parser.add_argument('--maxiter',default=20,type=int, help='Maximum number of iterations')
    parser.add_argument('--N',default=40,type=int, help='Parallelization Level')
    parser.add_argument('--seed',default=1234567,type=int, help='Seed used in random number generator')
    parser.add_argument('--output',default=None, help='If not None, cross validation is skipped, and U,V are trained over entire dataset and store it in files output_U and output_V')

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)

    return parser.parse_args()     

def train(args, sc):
    folds = {}

    if args.output is None:
        for k in range(args.folds):
            folds[k] = readRatings(args.data+"/fold"+str(k),sc)
    else:
        folds[0] = readRatings(args.data,sc)

    cross_val_rmses = []
    for k in folds:
        train_folds = [folds[j] for j in folds if j is not k ]  # excludes one fold for training

        if len(train_folds)>0:
            train = train_folds[0] 
            for fold in  train_folds[1:]:
                train=train.union(fold)  # combines all training folds
            train.repartition(args.N).cache()
            test = folds[k].repartition(args.N).cache()  # uses excluded fold for testing
            Mtrain=train.count()
            Mtest=test.count()

            print("Initiating fold %d with %d train samples and %d test samples" % (k,Mtrain,Mtest) )
        else:
            train = folds[k].repartition(args.N).cache()
            test = train
            Mtrain=train.count()
            Mtest=test.count()
            print("Running single training over training set with %d train samples. Test RMSE computes RMSE on training set" % Mtrain )

        i = 0
        change = 1.e99
        obj = 1.e99

        #Generate user profiles
        U = generateUserProfiles(train,args.d,args.seed,sc,args.N).cache()
        V = generateItemProfiles(train,args.d,args.seed,sc,args.N).cache()

        print "Training set contains %d users and %d items" %(U.count(),V.count())

        start = time()
        gamma = args.gain

        while i<args.maxiter and change > args.epsilon:
            i += 1

            joinedRDD = joinAndPredictAll(train,U,V,args.N).cache()

            oldObjective = obj
            obj = SE(joinedRDD) + normSqRDD(U,args.lam) + normSqRDD(V,args.lam)         
            change = np.abs(obj-oldObjective) 

            testRMSE = np.sqrt(1.*SE(joinAndPredictAll(test,U,V,args.N))/Mtest)

            gamma = args.gain / i**args.power
            U.unpersist()
            V.unpersist()
            U = adaptU(joinedRDD,gamma,args.lam,args.N).cache()
            V = adaptV(joinedRDD,gamma,args.mu,args.N).cache()

            now = time()-start
            print "Iteration: %d\tTime: %f\tObjective: %f\tTestRMSE: %f" % (i,now,obj,testRMSE)

            joinedRDD.unpersist()

        cross_val_rmses.append(testRMSE)

        train.unpersist()
        test.unpersist()

    if args.output is None:
       print "%d-fold cross validation error is: %f " % (args.folds, np.mean(cross_val_rmses))
    else:
       print "Saving U and V RDDs"
       U.saveAsTextFile(args.output+'_U')
       V.saveAsTextFile(args.output+'_V')
    return U, V, np.mean(cross_val_rmses)

SparkContext.setSystemProperty('spark.executor.memory', '100g')
SparkContext.setSystemProperty('spark.driver.memory', '100g')
# try:
#     sc = SparkContext.getOrCreate()
# except:
#     SparkContext.stop(sc)
sc = SparkContext(appName='Parallel MF')
# spark = SparkSession(sc)
sc.setLogLevel("ERROR")   

# SparkContext.stop(sc)
# SparkContext.setSystemProperty('spark.executor.memory', '100g')
# SparkContext.setSystemProperty('spark.driver.memory', '100g')
# sc = SparkContext("spark://10.99.248.66:7077", appName='Parallel MF')

epsilon = 1e-99 
outputfile = None
N = 40
seed = 1234567 
output = None 
verbose = False

fold_nums = 5
gain = 0.00025
data_name = '../data/beer'
power = 0.15
maxiter = 40

results = []
for lam in range(0, 2):
    for mu in range(0, 2):
        for d in range(1, 14):
            print('---------------------------------------')
            print('lam: {}, mu: {}, d: {}'.format(lam, mu, d))
            args = get_args(data_name, fold_nums, gain, power, epsilon, 
                             lam, mu, d, outputfile, maxiter, N, seed, 
                             output, verbose)
            U, V, rms = train(args, sc)
            results.append((lam, mu, d, rms))
            with open('../results/mf/beer.pickle', 'wb+') as f:
                pickle.dump(results, f)
            with open('../results/mf/beer.txt', 'ab+') as f:
                f.write('{} {} {} {}\n'.format(lam, mu, d, rms))