"""."""
import pandas as pd
import numpy as np
import glob

from config import features, path, raw_path, label


def select_merge():
    """Select feature we need, and merge."""
    data_zip = glob.glob(raw_path + '*.csv')
    print(data_zip)
    result = pd.DataFrame()
    for data in data_zip:
        print("Working on ", data)
        df = pd.read_csv(data)
        df = df[features + label]
        print("DataFrame: ", df.columns.values)
        df['price'] = df['price'].apply(lambda x: x.replace('$', '')).apply(lambda x: x.replace(',', '')).astype(np.float64)
        print(df['price'])

        result = result.append(df)
        print(result.shape)
    result.to_csv(path + 'cleanedlistings.csv', index=False)


def main():
    """."""
    select_merge()

if __name__ == '__main__':
    main()
