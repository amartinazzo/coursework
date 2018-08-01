from train import pre_process_data, predict
import pickle

# load chosen model

model = pickle.load(open('rf.sav', 'rb'))

# predict

X_test, _ = pre_process_data('test.csv')
predict(model, X_test)