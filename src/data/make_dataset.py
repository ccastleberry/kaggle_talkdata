# -*- coding: utf-8 -*-
import click
import logging
import os
import gc
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import build_computed_features

DEBUG = True

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
    print(str(train_path))
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

    len_train = train_chunksize
    val_size = val_chunksize

    cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
    val_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
    test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
    train_iter = pd.read_csv(train_path, dtype=dtypes, usecols=cols, iterator=True, parse_dates=['click_time'])

    for i in range(1,4):
        print('*****    Round {}   *****'.format(i))
        print('Loading Train data')
        train_df = train_iter.get_chunk(train_chunksize)

        print('Loading Validation data')
        val_df = pd.read_csv(train_path, dtype=dtypes, usecols=val_cols, skiprows=range(1, 150000000),
                            parse_dates=['click_time'], nrows=val_chunksize)
        train_df = train_df.append(val_df)
        del val_df; gc.collect()

        print('Loading Test data')
        if DEBUG:
            test_df = pd.read_csv(test_path, dtype=dtypes, usecols=test_cols, parse_dates=['click_time'], nrows=1000)
        else:
            test_df = pd.read_csv(test_path, dtype=dtypes, usecols=test_cols, parse_dates=['click_time'])
        train_df=train_df.append(test_df)
        del test_df; gc.collect()

        print('Building features.')
        train_df = build_computed_features.transform_data(train_df)
        print('Features built!')
        print('Splitting and saving')
        test_df = train_df[len_train:]
        val_df = train_df[(len_train-val_size):len_train]
        train_df = train_df[:(len_train-val_size)]

        train_filename = os.path.join(output_filepath, 'train{}.csv'.format(i))
        train_df.to_csv(train_filename, index=False)
        val_filename = os.path.join(output_filepath, 'val{}.csv'.format(i))
        val_df.to_csv(val_filename, index=False)
        test_filename = os.path.join(output_filepath, 'test{}.csv'.format(i))
        test_df.to_csv(test_filename, index=False)

        del test_df, val_df, train_df; gc.collect()





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
