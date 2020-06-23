import itertools
from datetime import datetime

import numpy
import pandas
import scikitplot
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from scipy import stats
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import KFold  # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
try:
    from sksurv.nonparametric import kaplan_meier_estimator
except ImportError:
    kaplan_meier_estimator = None
import random as rn

NUMERICALS = ['GIAGE1', 'HA_HEIGHT', 'HA_WEIGHT', 'TUDRPRWK', 'B1FND', 'HA_SMOKE_0.0', 'HA_SMOKE_1.0', 'HA_SMOKE_2.0', 'GIERACE_1.0',
              'GIERACE_2.0', 'GIERACE_3.0', 'GIERACE_4.0', 'GIERACE_5.0', 'CLINIC_1.0','CLINIC_2.0', 'CLINIC_3.0',  
              'CLINIC_4.0', 'CLINIC_5.0', 'CLINIC_6.0','NFWLKSPD_0.0', 'NFWLKSPD_1.0', 'NFWLKSPD_2.0']

# fix random seed for reproducibility
seed = 7
#
# # The below is necessary for starting Numpy generated random numbers
# # in a well-defined initial state.
numpy.random.seed(seed)
# # The below is necessary for starting core Python generated random numbers
# # in a well-defined state.
rn.seed(seed)

# fill numerical empty cells with median of the column and race with 1 (for white) and other categorical empty cells with 0
def fill_empty_cell(sample, attribute, data):
    if pandas.isnull(sample[attribute]):
        return data[attribute].median() if attribute in NUMERICALS else int(attribute == 'GIERACE')
    elif attribute == 'FRAC':
        return int(sample['FAANYSLD'] or sample['FAANYWST'] or sample['FAANYHIP'] or sample['XMDSQGE1'])
    elif attribute == 'STATUS':
        return int(sample['EFSTATUS'] > 0) or int(sample['FAANYSLD'] or sample['FAANYWST'] or sample['FAANYHIP'])
    elif attribute == 'DAYS':
        return min(sample['FAHIPFV1'], sample['FASLDFV1'], sample['FAWSTFV1'])
    else:
        return sample[attribute]


# weight
# def load_weight(sheet):
#    weight = []
#    df = pandas.read_excel('Estrada_63.xlsx', sheet_name=sheet)
#    w = list(df)[-1]
#    for i in range(len(df)):
#        weight.append(df.iloc[i][w])
#    return weight


# def weighted_grs(sample, data, weight):
# 	s = sample[list(data)[15:-6]].squeeze()
# 	for i in range
# 	return s.dot(weight)


# confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=pyplot.cm.Blues):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    pyplot.imshow(cm, interpolation='nearest', cmap=cmap)
    pyplot.title(title)
    pyplot.colorbar()
    tick_marks = numpy.arange(len(classes))
    pyplot.xticks(tick_marks, classes, rotation=45)
    pyplot.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pyplot.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')
    pyplot.tight_layout()


# Grid search for determining hyperparameters
# Hyper parameter tuning
def RandomSearch(estimator, modelName, params, Xtrain, ytrain, Xtest, ytest, score, verb=0):
    t0 = datetime.now()
    print('\nSearching grid -', modelName, '(' + score + ')...\n')
    grid = RandomizedSearchCV(estimator, param_distributions=params, cv=KFold(n_splits=3, shuffle=True), scoring=score, n_jobs=1, verbose=verb)  # n_jobs threads it if possible.
    grid.fit(Xtrain, ytrain)
    print('The best parameters are\n', grid.best_params_)
    print('The best', score, 'score is %0.4f \n' % (grid.best_score_ * 100))
    ypred = grid.predict(Xtest)
    # cm = confusion_matrix(ytest, ypred)
    numpy.set_printoptions(precision=2)
    # pyplot.figure()
    # plot_confusion_matrix(cm, classes=[0, 1], normalize=False, title='Normalized confusion matrix')
    # pyplot.show()
    print('\n', (datetime.now() - t0).total_seconds(), 'seconds')
    return grid.best_estimator_


# calculate confidence interval from the scores
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * numpy.array(data)
    n = len(a)
    m, se = numpy.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def get_data(plot=True):
    data = pandas.read_excel('mros_1103snps.xlsx')
    # drop HA_SLDFXFU where only 10% is filled, drop subjectid,
    data.drop(['HA_SLDFXFU', 'TURSMOKE', 'HA_SLDFX', 'HA_WRSTFX'], axis=1, inplace=True)
    # add genetic scores
    #sheet = 'c1_4321_GRS'
    #grs_data = pandas.read_excel('MrOS_Genotype_Genetic_Score(2018-12-12).xlsx', sheet_name=sheet)
    #sig_dict = {}
    #for i in range(len(grs_data['ID'])):
    #    key = grs_data['ID'].iloc[i]
    #    value = sig_dict[key] = grs_data['#ALLELE'].iloc[i]
        # change allele to number
    #    data[key] = data.apply(lambda sample: sample[key].count(value), axis=1)
    data['FRAC'] = 0
    data['STATUS'] = 0
    data['DAYS'] = 0
    # make the fractures into 1 variable
    for attribute in data.keys():
        data[attribute] = data.apply(lambda sample: fill_empty_cell(sample, attribute, data), axis=1)
    # drop the other fractured values
    #data.drop(['FAANYSLD', 'FAANYWST', 'FAANYHIP', 'XMDSQGE1', 'XMSQGE2', 'EFSTATUS', 'FAHIPFV1', 'FASLDFV1', 'FAWSTFV1'], axis=1, inplace=True)
    # encode the categorical data
    data = pandas.DataFrame(pandas.get_dummies(data, columns=['GIERACE', 'PHYS_MROS', 'NFWLKSPD']))
    # setting Y and X
    Y = data['FRAC']
    X = pandas.read_excel('norma_continu_var.xlsx')  
    #X_df = data.drop(['SUBJECTID', 'HA_LSD', 'BUAMEAN', 'FAHIPFV1', 'FASLDFV1', 'FAWSTFV1', 'EFSTATUS', 'HA_BMI', 'FAANYHIP', 'HA_CALCIUM', 'XMDSQGE1', 'XMSQGE2', 'CLINIC', 'FRAC', 'FAANYSLD', 'FAANYWST', 'STATUS', 'DAYS', 'FAANYSLD','FAANYWST'], axis=1)

    # weight_LS = load_weight('LS_sex-combined_beta')
    #features = list(data)[13:-8]
    #feature_data = data[features]
    # weight_LS = pandas.DataFrame(pandas.Series(weight_LS, index=features, name=0))
    #weight_FN = load_weight('FN_sex-combined_beta')
    # weight_FN == pandas.DataFrame(pandas.Series(weight_FN, index=features, name=0))
    # X_df['GRS_LS'] = feature_data.dot(weight_LS)
    #X_df['GRS_FN'] = feature_data.dot(weight_FN)
    #X_df.drop(features, axis=1, inplace=True)
    # if kaplan_meier_estimator is not None and plot:
    #     survival = numpy.array(X_df.apply(lambda sample: (sample['STATUS'], sample['DAYS']), axis=1), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    #     time, survival_prob = kaplan_meier_estimator(survival['Status'], survival['Survival_in_days'])
    #     pyplot.step(time, survival_prob, where="post")
    #     pyplot.ylabel(r'est. probability of survival $\hat{S}(t)$')
    #     pyplot.xlabel('time $t$ in days')
    #X_df.drop(['STATUS', 'DAYS', 'FRAC'], axis=1, inplace=True)
    # weight=pandas.Series(weight)
    # X_df['grs']=0
    # X_df['grs'] = X_df.apply(lambda sample: weighted_grs(sample, data, weight), axis=1)
    # print(list(X_df))
    # scale numerical entries to 0-1
    # numericals = ['GIAGE1', 'HA_HEIGHT', 'HA_WEIGHT', 'HA-SMOKE', 'GIERACE', 'TUDRPRWK', 'B1FND', 'GRS_FN']  # ['BUAMEAN', 'GIAGE1', 'HA_BMI', 'HA_CALCIUM', 'TUDRPRWK', 'GRS_FN', 'GRS_LS', 'B1FND', 'B1TLD', 'B1THD']  # , 'score1', 'score2', 'score3']
    #minMaxScaler = preprocessing.MinMaxScaler()
    #X_df[NUMERICALS] = minMaxScaler.fit_transform(X_df[NUMERICALS])
    # pca=PCA(n_components=10)
    # X_df=pca.fit_transform(X_df)
    # smote
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    sm = SMOTE(random_state=2, ratio = 1.0)
    x_train_s, y_train_s = sm.fit_resample(x_train, y_train)
    # Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, Y_df, test_size=0.2)
    parameters = {
        'loss': ['deviance', 'exponential'],
        'n_estimators': [200, 500, 800, 1000, 1200],
        'learning_rate': [0.001, 0.003, 0.005],
        'subsample': [0.3, 0.5, 0.7],  # <1.0 results in reduction of variance and increase in bias
        'min_samples_split': [2, 5, 8],
        'max_features': ['auto', 'log2', 'sqrt', 0.2],
        'random_state': [42],
        'max_depth': [2, 3, 5],
        'min_impurity_decrease': [0.15, 0.1, 0.08, 0.05]
    }
    # split data for parameter sweep
    model = RandomSearch(estimator=GradientBoostingClassifier(), modelName='Gradient Boosting Classifier', params=parameters, Xtrain=x_train_s, ytrain=y_train_s, Xtest=x_test, ytest=y_test, score='roc_auc')
    # model=GradientBoostingClassifier(subsample=0.3, n_estimators=800, min_samples_split=2, min_impurity_decrease=0.05, max_features='sqrt', max_depth=3, loss='deviance', learning_rate=0.01)
    model.fit(x_train_s, y_train_s)
    print(model.feature_importances_)
    # ypred = model.predict(Xtest)
    yscore_raw = model.predict_proba(x_test)
    yscore = [s[1] for s in yscore_raw]
    fpr, tpr, thresh = roc_curve(y_test, yscore)
    auc = roc_auc_score(y_test, yscore)
    ytest = numpy.array(y_test)
  # yscore = numpy.array(yscore)
    # n_bootstraps = 1000
    # bootstrapped_scores = []
    # rng = numpy.random.RandomState()
    # for i in range(n_bootstraps):
    #     # bootstrap by sampling with replacement on the prediction indices
    #     indices = rng.random_integers(0, len(yscore) - 1, len(yscore))
    #     if len(numpy.unique(ytest[indices])) < 2:
    #         # We need at least one positive and one negative sample for ROC AUC to be defined: reject the sample
    #         continue
    #     bootstrapped_scores.append(roc_auc_score(ytest[indices], yscore[indices]))
    # print(mean_confidence_interval(bootstrapped_scores))
    # plot roc curve
    if plot:
        y_probas = model.predict_proba(x_test)  # predicted probabilities generated by sklearn classifier
        scikitplot.metrics.plot_roc(ytest, y_probas, plot_macro=False, plot_micro=False, classes_to_plot=[1], title='ROC Curve by Gradient Boosting Model')
        pyplot.show()
    return fpr, tpr, thresh, auc


if __name__ == '__main__':
    get_data()
