import pandas as pd
import lightgbm as lgb
import os
import gc

BASE_PATH = os.path.join('..', '..')
DATA_FOLDER = os.path.join(BASE_PATH , 'data', 'processed')
TRAIN_PATH = os.path.join(DATA_FOLDER, 'train{}.csv')
VAL_PATH = os.path.join(DATA_FOLDER, 'val{}.csv')
TEST_PATH =  os.path.join(DATA_FOLDER, 'test{}.csv')
SUBMISSION_FOLDER = os.path.join(BASE_PATH, 'submissions')
SUB_VAL_FOLDER = os.path.join(SUBMISSION_FOLDER, 'validation')
SUB_TEST_FOLDER = os.path.join(SUBMISSION_FOLDER, 'test')
VAL_SUB_PATH = os.path.join(SUB_VAL_FOLDER, 'cf_lgb_val_subs.csv')
TEST_SUB_PATH = os.path.join(SUB_TEST_FOLDER, 'cf_lgb_test_subs.csv')



dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    #'click_id'      : 'uint32'
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

target = 'is_attributed'
predictors = ['app', 'channel', 'click_id', 'device', 'ip', 'os',
       'hour', 'day', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8',
       'ip_tcount', 'ip_app_count', 'ip_app_os_count', 'ip_tchan_count',
       'ip_app_os_var', 'ip_app_channel_var_day', 'ip_app_channel_mean_hour',
       'ip_app_channel_mean_attrib', 'nextClick', 'nextClick_shift']

categorical_features = ['app', 'channel', 'click_id', 'device', 'ip', 'os']

'''
xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                        feature_name=predictors,
                        categorical_feature=categorical_features
                        )
'''

evals_results = {}
early_stopping_rounds = 30
num_boost_round = 1000

val_sub_df = pd.DataFrame()
test_sub_df = pd.DataFrame()

# Loop Through all three training sets
for i in  range(1,4):
    print('***** Round 1 *****')
    print("Reading in training file.")
    train_df = pd.read_csv(TRAIN_PATH.format(i), dtype=dtypes)

    # build lgb dataset
    xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                        feature_name=predictors,
                        categorical_feature=categorical_features
                        )

    # Read in val set
    val_df_early = pd.read_csv(VAL_PATH.format(i))

    xgval = lgb.Dataset(val_df_early[predictors].values, label=val_df_early[target].values,
                        feature_name=predictors,
                        categorical_feature=categorical_features
                        )
    # Train Model

    bst1 = lgb.train(lgb_params,
                     xgtrain,
                     valid_sets=[xgtrain, xgval],
                     valid_names=['train', 'val'],
                     evals_result=evals_results,
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10
                     )
    best_iter = bst1.best_iteration


    # Train val set

    # loop through validation sets:
    for j in range(1,4):
        
        val_df = pd.read_csv(VAL_PATH.format(i))
        
        # get predicitons
        val_preds = bst1.predict(val_df[predictors],num_iteration=best_iter)
        del val_df; gc.collect()

        # add validation predictions
        if j ==1:
            val_subs = val_preds / 3.0
        else:
            val_subs += val_preds / 3.0

    # record val predictions
    if i == 1:
        val_sub_df['cf_lgb_subs'] = val_subs / 3.0
    else:
        val_sub_df['cf_lgb_subs'] = val_sub_df['cf_lgb_subs'] + (val_subs / 3.0)

    # Train Test set

    # loop through test sets:

    for j in range(1,4):
        # Read in test set
        test_df = pd.read_csv(TEST_PATH.format(i))

        # get predictions
        test_preds = bst1.predict(test_df[predictors], num_iteration=best_iter)
        del test_df; gc.collect()

        # add test predictions
        if j ==1:
            test_subs = test_preds / 3.0
        else:
            test_subs += test_preds / 3.0
    if i == 1:
        test_sub_df['cf_lgb_subs'] = test_subs / 3.0
    else:
        test_sub_df['cf_lgb_subs'] = test_sub_df['cf_lgb_subs'] + (test_subs / 3.0)

    # Record test predictions


# Save predictions
val_sub_df.to_csv(VAL_SUB_PATH, index=False)
test_sub_df.to_csv(TEST_SUB_PATH, index=False)