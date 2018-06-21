from kaggle import pre_process_data
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X_test, _ = pre_process_data('test.csv')

# load chosen model

model = pickle.load(open('adaboost.sav', 'rb'))

# predict

y_hat = model.predict(X_test)

# save to csv

output = np.zeros((X_test.shape[0], 2))
output[:,0] = X_test[:,0]
output[:,1] = y_hat[0]

df = pd.DataFrame(output, columns=['id','y'], dtype='int')

print(df.head(5))
df.to_csv('submission.csv', index=False)