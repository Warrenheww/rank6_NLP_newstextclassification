############################### LightBGM Voting #######################################

import numpy as np
import pandas as pd
import logging
#from sklearn.externals import joblib
import joblib
np.warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
import lightgbm as lgb

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

logging.basicConfig(filename='train.log', level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

# load data set
train_df = pd.read_csv('./train_set.csv', sep='\t')
test_df = pd.read_csv('./test_a.csv', sep='\t', nrows=None)


# feature vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=50000)
vectorizer.fit(np.concatenate((train_df['text'].iloc[:].values,test_df['text'].iloc[:].values),axis=0))

train_word_features = vectorizer.transform(train_df['text'].iloc[:].values)
test_word_features = vectorizer.transform(test_df['text'].iloc[:].values)


# parameters
params={}
params['n_estimators']=500
params['subsample']=0.72
params['colsample_bytree']=0.599
params['reg_alpha']=0.001
params['reg_lambda']=0.5
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='multiclass' #Multi-class target feature
params['metric']='multi_logloss' #metric for multi-class
params['learning_rate']=0.088
params['max_depth']=100
params['num_leaves']=67
params['min_child_samples']=21
params['num_class']=14
params['n_jobs']=-1 


# train model
X_train = train_word_features
y_train = train_df['label']
X_test = test_word_features

KF = KFold(n_splits=6, random_state=1) 

# save the test result
test_pred = np.zeros((X_test.shape[0], 1), int)  
for KF_index, (train_index,valid_index) in enumerate(KF.split(X_train)):
    
    logging.info("The No. {} cross validation begines...".format(KF_index+1))

    # divide train and valid dataset
    x_train_, x_valid_ = X_train[train_index], X_train[valid_index]
    y_train_, y_valid_ = y_train[train_index], y_train[valid_index]

    # fit model
    d_train=lgb.Dataset(x_train_, label=y_train_)
    d_val = lgb.Dataset(x_valid_, label=y_valid_, reference=d_train) 
    clf=lgb.train(params,d_train,num_boost_round=1000,valid_sets=d_val,early_stopping_rounds=100)
    
    # predict
    val_pred = np.argmax(clf.predict(x_valid_, num_iteration=clf.best_iteration),axis=1)
    
    logging.info("The F1 Score is：{}".format(f1_score(y_valid_, val_pred, average='macro')))
    
    # save the test result 对结果编号
    test_pred = np.column_stack((test_pred, np.argmax(clf.predict(X_test,num_iteration=clf.best_iteration),axis=1)))
    
    #save models
    logging.info("Saving model...")
    joblib.dump(clf, f'./model/my_LGBM_model_5cv_{KF_index}.pkl', compress=3)

# get the final result according to the most votes
logging.info("test_pred.shape: {}".format(test_pred.shape))
logging.info("The first column of test_pred is pure [0], removing...")

test_pred=test_pred[...,1:test_pred.shape[1]]

logging.info("test_pred.shape: {}".format(test_pred.shape))

preds = []
for i, test_list in enumerate(test_pred):
    preds.append(np.argmax(np.bincount(test_list)))
preds = np.array(preds)

# store the final results
df = pd.DataFrame()
df['label'] = preds
df.to_csv('./dalma_lgbm.csv', index=False)
