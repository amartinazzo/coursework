from train import pre_process_data
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X_test, _ = pre_process_data('test.csv')

# load chosen model

model = pickle.load(open('rf_gridsearch.sav', 'rb'))

# predict

# prob = model.predict_proba(X_test)
# print(prob[0:20,:])
y_hat = model.predict(X_test)

# save to csv

output = np.zeros((X_test.shape[0], 2))
output[:,0] = X_test[:,0]
output[:,1] = y_hat

df = pd.DataFrame(output, columns=['id','y'], dtype='int')

df.to_csv('submission_gridsearch.csv', index=False)