"""Train.py."""
from dataprepare import read_data, clean_data
# from aquablu.prepare import aggregate_data
import argparse

from model import model
from task_configs import raw_features, features, label, spark
from pyspark.sql import functions as f


def run_task(data):
    """."""
    ml_model = model(label=label,
                     features=features,
                     raw_features=raw_features,
                     standardize=(False, None),
                     nomralized=False,
                     task_type='regression',
                     params=None,
                     category=True,
                     cross_validation=True,
                     ml_algo='gbtregressor',
                     split_ratio=[0.7, 0.3])
    predictions = ml_model.train_pipeline(data)
    return predictions


def analysis(predictions):
    """."""
    # predictions.select(raw_features + [label, '[prediction']]).write.csv('../../data/airbnb_predictions_csv', header=True)
    # predictions.select(raw_features + [label, 'prediction']).write.json('../../data/airbnb_predictions_json')
    byneighbors = predictions.groupby('neighbourhood').agg(f.collect_list('prediction'),
                                                           f.collect_list('last_scraped'))
    byneighbors.show()
    byneighbors.write.json('../data/airbnb_predictions_by_neighborhood_avg_json')
    return


def parse_argument():
    """Hyperparmeter tuning."""
    ap = argparse.ArgumentParser()
    ap.add_argument("-airbnb", "--airbnb_path",
                    default='../data/cleanairbnb.csv')

    return vars(ap.parse_args())


def main(args):
    """."""
    # aggregate_data('./', csv_cols=["price", "accommodates",
    #                                "bedrooms", "beds",
    #                                "bathrooms", "review_scores_rating",
    #                                "property_type", "room_type"],
    #                destnation=args['airbnb_path'])

    airbnb = read_data(args['airbnb_path'], spark)
    airbnb = clean_data(airbnb)
    predictions = run_task(airbnb)
    analysis(predictions)


if __name__ == '__main__':
    args = parse_argument()
    main(args)