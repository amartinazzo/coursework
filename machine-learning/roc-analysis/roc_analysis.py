from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interp
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

def pre_process_data(filename):
    print('\npre-processing ', filename)

    features = pd.read_csv(filename)
    print('original shape: ', features.shape)

    features = features[['class', 'habitat', 'cap-color']]

    features = pd.get_dummies(features,
        prefix=features.columns.values.tolist(),
        drop_first=True)
    print('final shape: ', features.shape)
    # features.to_csv('{}_one_hot.csv'.format(filename), index=False)

    labels = np.array(features['class_p'])
    features = features.drop(['class_p'], axis=1)

    print('final shape: ', features.shape)

    return np.array(features), labels


def generate_curves(cv, classifier, label, plot_color):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # print(probas_)
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plt.plot(fpr, tpr, lw=1, alpha=0.3,
        #          label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        plt.plot(fpr, tpr, lw=1, alpha=0.4, color=plot_color)
        i += 1

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color=plot_color,
             label=label+' (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=plot_color, alpha=0.1)


# main

X, y = pre_process_data('mushrooms.csv')

cv = StratifiedKFold(n_splits=10)
weights={1:10}
tree = DecisionTreeClassifier(class_weight=weights)
lr = LogisticRegression(class_weight=weights)

generate_curves(cv, tree, 'Decision Tree', 'r')
generate_curves(cv, lr, 'Logistic Regression', 'b')

plt.plot([0, 1], [0, 1], linestyle='--', alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves')
plt.legend(loc="lower right")
plt.show()
