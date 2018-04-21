# -*- coding: utf-8 -*-
import click
import logging
import os
import gc
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
#import pandas as pd
import ray.dataframe as pd
import build_computed_features_ray

DEBUG = True
val_size = 20000000

#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    train_path = os.path.join(input_filepath, 'train.csv')
    test_path = os.path.join(input_filepath, 'test.csv')

    dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
    if DEBUG:
        train_chunksize = 10000
        val_chunksize = 1000
    else:
        train_chunksize = 50000000
        val_chunksize = 10000000

    cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
    test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
    train_df = pd.read_csv(train_path, dtype=dtypes, usecols=cols, parse_dates=['click_time'])
    test_df = pd.read_csv(test_path, dtype=dtypes, usecols=test_cols, parse_dates=['click_time'])
    len_train = train_df.shape[0]
    train_df=train_df.append(test_df)

    train_df = build_computed_features_ray.transform_data(train_df)

    test_df = train_df[len_train:]
    val_df = train_df[(len_train-val_size):len_train]
    train_df = train_df[:(len_train-val_size)]
    gc.collect()

    
    train_filename = os.path.join(output_filepath, 'train{}'.format(i))
    train_df.to_csv(train_filename, index=False)
    val_filename = os.path.join(output_filepath, 'val{}'.format(i))
    val_df.to_csv(val_filename, index=False)
    test_filename = os.path.join(output_filepath, 'test{}'.format(i))
    test_df.to_csv(test_filename, index=False)





if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    base_path = os.path.join('..', '..')
    input_filepath = os.path.join(base_path, 'data', 'raw')
    output_filepath = os.path.join(base_path, 'data', 'processed')
    main(input_filepath, output_filepath)
