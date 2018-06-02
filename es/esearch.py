from elasticsearch import Elasticsearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from config import spark_path, AWS_ACCESS_KEY
from config import AWS_SECRET_KEY, region, service, host, INDEX_NAME
from pprint import pprint

import pandas as pd
import json
import glob

awsauth = AWS4Auth(AWS_ACCESS_KEY, AWS_SECRET_KEY, region, service)

es = Elasticsearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)


def transmit(files):
    """Transform from csv to json."""
    id = 1
    bulk_file = ''
    for file_obj in files:
        for line in open(file_obj, mode="r"):
            document = json.loads(line)
            # If running this script on a website, you probably need to prepend the URL and path to html_file.
            bulk_file += '{ "index" : { "_index" : "prediction_price", "_type" : "byneighbors", "_id" : "' + str(id) + '" } }\n'

            # The optional_document portion of the bulk file
            bulk_file += json.dumps(document) + '\n'
            id += 1

    es.bulk(bulk_file)
    print(es.get(index="prediction_price", doc_type="byneighbors", id="*"))
    return id


def main():
    """."""
    files = glob.glob(spark_path + '*.json')
    print("Files = {}".format(len(files)))
    transmit(files)


if __name__ == '__main__':
    main()
