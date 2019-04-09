# ParallelProcessingFinal
Final Project for EECE 5698

## Current ALS
Assumes input files are stored in folds, ex data/fold0, data/fold1.  
Each file should be stored as a CSV with fields (user, item, rating)  

#### ALSRDD.py
Performs a collaborative filtering approach without knowledge of item attributes using RDDs  

#### ALSDataFrame.py
Performs a collaborative filtering approach without knowledge of item attributes using Dataframes  

## Current Preprocessing
#### preprocessBeer.py
Create a mapping file between beerId and beerName
Create a dataframe with brewery/non-numerical information removed
Saves a 5 fold version of beer data in data directory

## Notes
#### Mllib vs Ml
Mllib is deprecated and focuses on RDD machine learning tasks  
ml deals with pipelines and Dataframes  
Mllib is slowly being deprecated  

## Questions
What was the motivation for partitioning with 40 partitions in the last assignment?  
Cold start strategy --> drop, how does this change RMSE value? Could explain different RMSE's for df and non-df...  
What could we do with brewery name, beer_style, beer_name?? (non-numerical vals)  
Preprocessing, remove all beers with no abv?  
Include review overall in the closed form portion of the project?  
Does a cache persist beyond the scope of a function?  

## Links  
[docs](https://spark.apache.org/docs/2.3.2/api/python/pyspark.ml.html#pyspark.ml.recommendation.ALS)  
[example](https://spark.apache.org/docs/2.3.2/ml-collaborative-filtering.html)  
[src](https://spark.apache.org/docs/2.3.2/api/python/_modules/pyspark/ml/recommendation.html#ALS)  