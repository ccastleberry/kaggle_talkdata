'''
This file will load the test data set and create an sklearn one hot encoder from it and then save that model
'''

# script parameters
DEBUG = False

#imports
import os
import pandas as pd
import gc
import pickle
from sklearn.preprocessing import OneHotEncoder


# setup paths
PROJECT_PATH = os.path.join('..', '..')
path_test_data = os.path.join(PROJECT_PATH, 'data', 'raw', 'test.csv')
path_one_hot_pickle = os.path.join(PROJECT_PATH, 'models', 'preprocessing', 'one_hot_encoder.pickle')

# load test data
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

cols = ['ip', 'app', 'device', 'os', 'channel']

print('Loading data...')
if DEBUG:
    test_df = pd.read_csv(path_test_data, usecols=cols, dtype=dtypes, nrows=100)
    print(test_df.head())
else:
    test_df = pd.read_csv(path_test_data, usecols=cols, dtype=dtypes)

print('Converting to np array...')
X_test = test_df.values

del test_df
gc.collect()

# build one hot encoder
ohe = OneHotEncoder()
print('Fitting Encoder...')
ohe.fit(X_test)

print('Saving Encoder...')
with open(path_one_hot_pickle, 'wb') as f:
    pickle.dump(ohe, f)

print('Complete!')

