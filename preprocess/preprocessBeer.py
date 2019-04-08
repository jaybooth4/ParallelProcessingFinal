import pandas as pd

DATAFILE = "../data/beer_reviews.csv"

def main():
    df = pd.read_csv(DATAFILE)
    beerLookup = df.loc[:, ['beer_beerid', 'beer_name']].copy()
    beerLookup.drop_duplicates(inplace=True)
    print beerLookup
    df.drop(['brewery_id', 'review_time', 'brewery_name', 'beer_style', 'beer_name'], axis=1, inplace=True)
    print df
    print df.columns
    print df.describe()

if __name__ == "__main__":
    main()