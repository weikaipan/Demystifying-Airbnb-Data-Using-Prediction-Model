"""Train.py."""
from dataprepare import read_data, clean_data
# from aquablu.prepare import aggregate_data
import argparse

from model import model
from task_configs import features, label, spark


def run_task(data):
    """."""
    ml_model = model(label=label,
                     features=features,
                     standardize=(False, None),
                     nomralized=False,
                     task_type='regression',
                     params=None,
                     category=True,
                     cross_validation=True,
                     ml_algo='gbtregressor',
                     split_ratio=[0.7, 0.3])
    ml_model.train_pipeline(data)
    return


def parse_argument():
    """Hyperparmeter tuning."""

    ap = argparse.ArgumentParser()
    ap.add_argument("-airbnb", "--airbnb_path",
                    default='../../data/cleanairbnb.csv')

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
    run_task(airbnb)


if __name__ == '__main__':
    args = parse_argument()
    main(args)