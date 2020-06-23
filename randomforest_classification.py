# Random forest

import itertools
import time

import numpy
import pandas
import scikitplot
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from scipy import stats
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, brier_score_loss, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold  # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import random as rn

NUMERICALS = ['GIAGE1', 'HA_HEIGHT', 'HA_WEIGHT', 'TUDRPRWK', 'B1FND', 'GRS_FN', 'HA_SMOKE', 'GIERACE_1.0','GIERACE_2.0', 'GIERACE_3.0', 'GIERACE_4.0', 'GIERACE_5.0', 'NFWLKSPD_0.0', 'NFWLKSPD_1.0', 'NFWLKSPD_2.0']

# fix random seed for reproducibility
seed = 7
#
# # The below is necessary for starting Numpy generated random numbers
# # in a well-defined initial state.
numpy.random.seed(seed)
# # The below is necessary for starting core Python generated random numbers
# # in a well-defined state.
rn.seed(seed)

# fill numerical empty cells with median of the column, race with 1 (for white) and other categorical empty cells with the 0
def fill_empty_cell(sample, attribute, data):
    if pandas.isnull(sample[attribute]):
        if attribute in NUMERICALS:
            return data[attribute].median()
        return int(attribute == 'GIERACE' )
    elif attribute == 'FRAC':
        return int(sample['FAANYSLD'] or sample['FAANYWST'] or sample['FAANYHIP'] or sample['XMDSQGE1'])
    return sample[attribute]


# Hyper parameter tuning
def RandomSearch(estimator, modelName, params, Xtrain, ytrain, Xtest, ytest, score, verb=0):
    t0 = time.time()
    print('\nSearching grid -', modelName, '(' + score + ')...\n')
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    grid = RandomizedSearchCV(estimator, param_distributions=params, cv=cv, scoring=score, n_jobs=1, verbose=verb)  # n_jobs threads it if possible.
    grid.fit(Xtrain, ytrain)
    numpy.set_printoptions(precision=2)
    return grid.best_estimator_


# calculate confidence interval from the scores
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * numpy.array(data)
    m, se = numpy.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., len(a) - 1)
    return m, m - h, m + h

def get_data(plot=True):
    data = pandas.read_excel('mros_1103snps.xlsx')
    # drop HA_SLDFXFU where only 10% is filled, drop subjectid,
    data.drop(['HA_SLDFXFU', 'TURSMOKE', 'HA_SLDFX', 'HA_WRSTFX'], axis=1, inplace=True)

    # make frac attribute as fracture and return the values from fill_empty_Cell with either FAANYSLD, FAANYWST or HIP
    data['FRAC'] = 0
    for attribute in data.keys():
        data[attribute] = data.apply(lambda sample: fill_empty_cell(sample, attribute, data), axis=1)
    # drop the other fractured values
    data.drop(['FAANYSLD', 'FAANYWST'], axis=1, inplace=True)
    # encode the categorical data
    data = pandas.DataFrame(pandas.get_dummies(data, columns=['GIERACE', 'PHYS_MROS', 'NFWLKSPD']))
    #features = list(data)[20:-6]
    print(data.shape, ' SHAPE OF DAT')
    # setting Y and X
    Y = data['FRAC']
    X = pandas.read_excel('norma_continu_var.xlsx')
    #X = data.drop(['SUBJECTID', 'HA_LSD', 'BUAMEAN', 'FAHIPFV1', 'FASLDFV1', 'FAWSTFV1', 'EFSTATUS',
    #                'HA_BMI', 'FAANYHIP', 'HA_CALCIUM', 'XMDSQGE1', 'XMSQGE2', 'CLINIC', 'FRAC'], axis=1)

    #feature_data = data[features]

    #X.drop(features, axis=1, inplace=True)
    # smote
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    sm = SMOTE(random_state=2, ratio = 1.0)
    
    x_train_s, y_train_s = sm.fit_resample(x_train, y_train)
    # Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, Y_df, test_size=0.2)
    parameters = {
        'n_estimators': [300, 800, 1000, 1200],
        'max_features': ['auto', 'sqrt', 0.2],
        'min_samples_split': [10, 20, 5, 2],
        'max_depth': [2, 3, 5, 8],
        'random_state': [45]
    }
    # split data for parameter sweep
    gbr = RandomForestClassifier()
    model = RandomSearch(estimator=gbr, modelName='Random Forest Classifier', params=parameters, Xtrain=x_train_s, ytrain=y_train_s, Xtest=x_test, ytest=y_test, score='roc_auc')
    # model=RandomForestClassifier(n_estimators=800, max_depth=5)
    model.fit(x_train_s, y_train_s)
    print(model.feature_importances_)
    # ypred = model.predict(Xtest)
    yscore_raw = model.predict_proba(x_test)
    yscore = [s[1] for s in yscore_raw]
    fpr, tpr, thresh = roc_curve(y_test, yscore)
    print(confusion_matrix(y_test,model.predict(x_test)))
    
    print(list(X))
    
#    print(confusion_matrix(ytest,yscore))
    auc = roc_auc_score(y_test, yscore)
    ytest = numpy.array(y_test)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(fpr, tpr, label='%s ROC (area = %0.3f)' % ('Random Forest', auc))

        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity(False Positive Rate)')
        plt.ylabel('Sensitivity(True Positive Rate)')
        plt.title('Receiver Operating Characteristic-Random Forest')
        plt.legend(loc="lower right")
        plt.savefig('random_forest_MOF.png')
        plt.show()
    return fpr, tpr, thresh, auc


if __name__ == '__main__':
    get_data()