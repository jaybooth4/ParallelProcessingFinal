import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

INPUT = "../data/small_data/fold0"

def main():
    pd.read_csv(INPUT)
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    print LinearRegression(fit_intercept=False).fit(X, y).coef_

if __name__=="__main__":
    main()