from datetime import datetime
from elasticsearch import Elasticsearch
from config import awses_end, spark_path
from pprint import pprint

import glob
import requests
import pandas as pd

es = Elasticsearch([awses_end])

def csv_to_json(obj):
    """Transform from csv to json."""
    columns = obj.columns
    


def main():
    """."""
    res = requests.get(awses_end)
    pprint(res.content)
    files = glob.glob(spark_path + '*.csv')
    print("Length of recent news: {}".format(len(files)))
    for f in files:
        obj = pd.read_csv(f)
        print(obj)


if __name__ == '__main__':
    main()
