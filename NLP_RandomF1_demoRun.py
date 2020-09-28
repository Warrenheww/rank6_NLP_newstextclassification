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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import OneHotEncoder

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

logging.basicConfig(filename='train.log', level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

# load data set
train_df = pd.read_csv('./train_set.csv', sep='\t')
test_df = pd.read_csv('./test_a.csv', sep='\t', nrows=None)


# feature vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
vectorizer.fit(np.concatenate((train_df['text'].iloc[:].values,test_df['text'].iloc[:].values),axis=0))

train_word_features = vectorizer.transform(train_df['text'].iloc[:].values)
test_word_features = vectorizer.transform(test_df['text'].iloc[:].values)

# label one-hot encoding
y_train=train_df['label'].iloc[:].values
n=y_train.shape[0]
train_label = y_train.reshape(n,1)
ohe = OneHotEncoder()
ohe.fit([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13]])
train_label_oh = ohe.transform(train_label).toarray()

# train model
logging.info("Training model...")
X_train = train_word_features
y_train = train_label_oh
X_test = test_word_features


clf = RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_split=2,
                             max_features=0.3,min_samples_leaf=1,random_state=32,
                             oob_score=True,verbose=100,n_jobs=-1)

KF = KFold(n_splits=5, random_state=1) 

# save the test result
test_pred = np.zeros((X_test.shape[0], 1), int)  
for KF_index, (train_index,valid_index) in enumerate(KF.split(X_train)):
    
    logging.info("The No. {} cross validation begines...".format(KF_index+1))

    # divide train and valid dataset
    x_train_, x_valid_ = X_train[train_index], X_train[valid_index]
    y_train_, y_valid_ = y_train[train_index], y_train[valid_index]

    # fit model
    clf.fit(x_train_, y_train_)
    logging.info("oob score is: {}".format(clf.oob_score_))
    
    # predict
    val_pred = clf.predict(x_valid_)
    preddd=[int(np.matrix(item)*np.matrix([0,1,2,3,4,5,6,7,8,9,10,11,12,13]).T) for item in val_pred]
    realll=[int(np.matrix(item)*np.matrix([0,1,2,3,4,5,6,7,8,9,10,11,12,13]).T) for item in y_valid_]
    
    logging.info("The F1 Score is：{}".format(f1_score(realll, preddd, average='macro')))
    
    # save the test result 对结果编号
    testtt=clf.predict(X_test)
    final=[int(np.matrix(item)*np.matrix([0,1,2,3,4,5,6,7,8,9,10,11,12,13]).T) for item in testtt]
    test_pred = np.column_stack((test_pred, final))
    
    #save models
    logging.info("Saving model...")
    joblib.dump(clf, f'./model/my_RFoh_model_5cv_{KF_index}.pkl', compress=3)

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
df.to_csv('./dalma_RFoh.csv', index=False)
