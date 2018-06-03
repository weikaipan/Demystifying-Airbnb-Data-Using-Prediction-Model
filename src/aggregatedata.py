"""."""
import pandas as pd
import glob

from config import features, path, raw_path


def select_merge():
    """Select feature we need, and merge."""
    data_zip = glob.glob(raw_path + '*.csv')
    print(data_zip)
    result = pd.DataFrame()
    for data in data_zip:
        print("Working on ", data)
        df = pd.read_csv(data)
        df = df[features]
        result.append(df)
    result.to_csv(path + 'cleanedlistings.csv')


def main():
    """."""
    select_merge()

if __name__ == '__main__':
    main()
