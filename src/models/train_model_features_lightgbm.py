import pandas as pd
import lightgbm as lgb
import os
import gc

BASE_PATH = os.path.join('..', '..')
DATA_FOLDER = os.path.join(BASE_PATH , 'data', 'processed')
SUBMISSION_FOLDER = os.path.join(BASE_PATH, 'submissions')


dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
    'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': 4,
    'verbose': 0,
    'metric': 'auc',
    'scale_pos_weight':200 # because training data is extremely unbalanced
}

xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                        feature_name=predictors,
                        categorical_feature=categorical_features
                        )
xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                        feature_name=predictors,
                        categorical_feature=categorical_features
                        )

evals_results = {}

bst1 = lgb.train(lgb_params, 
                    xgtrain, 
                    valid_sets=[xgtrain, xgvalid], 
                    valid_names=['train','valid'], 
                    evals_result=evals_results, 
                    num_boost_round=num_boost_round,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=10, 
                    feval=feval)