import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from xgboost import XGBClassifier
from WORC.statistics.delong import delong_roc_test
import math


def nested_cross_val(NUM_TRIALS, estimator, param_grid, X, y):
    # Arrays to store scores
    acc = np.zeros(NUM_TRIALS)
    roc = np.zeros(NUM_TRIALS)
    f1 = np.zeros(NUM_TRIALS)
    specificity = np.zeros(NUM_TRIALS)
    sensitivity = np.zeros(NUM_TRIALS)

    specificity_score = make_scorer(recall_score, pos_label=0)
    scoring = {"balanced_accuracy": "balanced_accuracy",
               "roc_auc": "roc_auc",
               "f1": "f1",
               "specificity": specificity_score,
               "sensitivity": "recall"}

    # Loop for each trial
    for i in tqdm(range(NUM_TRIALS)):
        # split data into k folds
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)
        outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)

        # Nested CV with parameter optimization
        clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=inner_cv, n_jobs=-1)
        scores = cross_validate(clf, X=X, y=y, cv=outer_cv, return_estimator=True, scoring=scoring)
        acc[i] = scores["test_balanced_accuracy"].mean()
        roc[i] = scores["test_roc_auc"].mean()
        f1[i] = scores["test_f1"].mean()
        specificity[i] = scores["test_specificity"].mean()
        sensitivity[i] = scores["test_sensitivity"].mean()

    print(f'Results for XGBoost on {str(X)}:')
    print("ACCURACY:", np.mean(acc), acc)
    print("ROC AUC:", np.mean(roc), roc)
    print("F1:", np.mean(f1), f1)
    print("Specificity:", np.mean(specificity), specificity)
    print("Sensitivity:", np.mean(sensitivity), sensitivity)
    return np.mean(acc), np.mean(roc), np.mean(f1), np.mean(specificity), np.mean(sensitivity)


def get_predictions_and_feature_importance(estimator1, estimator2, param_grid, X_rewards, y, X_norewards):
    test_indeces1 = []
    test_indeces2 = []
    ground_truths = []
    proba_preds1 = []
    proba_preds2 = []

    predicted_probs1 = {}
    predicted_probs2 = {}

    importances1 = []
    importances2 = []

    # split data into k folds
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True)
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True)

    # Loop for rewards classifier
    # Looping through the outer loop, feeding each training set into a GSCV as the inner loop
    for train_index,test_index in outer_cv.split(X_rewards, y):
        X_train = X_rewards.loc[X_rewards.index[train_index]]
        X_test = X_rewards.loc[X_rewards.index[test_index]]
        y_train = y.loc[y.index[train_index]]
        y_test = y.loc[y.index[test_index]]

        GSCV = GridSearchCV(estimator=estimator1,param_grid=param_grid,cv=inner_cv, scoring="roc_auc")
        GSCV.fit(X_train,y_train.values.ravel())

        # The best hyper parameters from GSCV is now being tested on the unseen outer loop test data.
        pred_proba = GSCV.predict_proba(X_test)

        # Appending the prediction scores of the "winning" model
        test_indeces1.extend(test_index)
        ground_truths.extend(y_test.values.ravel())
        proba_preds1.extend([x[1] for x in pred_proba])

        # Appending the feature importance scores of the "winning" model
        importances1.append(GSCV.best_estimator_.feature_importances_)

    predicted_probs1["test_index"] = test_indeces1
    predicted_probs1["ground_truth"] = ground_truths
    predicted_probs1["prediction_one"] = proba_preds1

    for train_index,test_index in outer_cv.split(X_norewards, y):
        X_train = X_norewards.loc[X_norewards.index[train_index]]
        X_test = X_norewards.loc[X_norewards.index[test_index]]
        y_train = y.loc[y.index[train_index]]
        y_test = y.loc[y.index[test_index]]

        GSCV = GridSearchCV(estimator=estimator2,param_grid=param_grid,cv=inner_cv, scoring="roc_auc")
        GSCV.fit(X_train,y_train.values.ravel())

        # The best hyper parameters from GSCV is now being tested on the unseen outer loop test data.
        pred_proba = GSCV.predict_proba(X_test)

        # Appending the prediction scores of the "winning" model
        test_indeces2.extend(test_index)
        proba_preds2.extend([x[1] for x in pred_proba])

        # Appending the feature importance scores of the "winning" model
        importances2.append(GSCV.best_estimator_.feature_importances_)

    predicted_probs2["test_index"] = test_indeces2
    predicted_probs2["prediction_two"] = proba_preds2

    prediction1 = pd.DataFrame.from_dict(predicted_probs1)
    prediction2 = pd.DataFrame.from_dict(predicted_probs2)

    predictions = pd.merge(prediction1, prediction2, on="test_index")

    columns = X_rewards.columns
    importances1 = [sum(x) / len(x) for x in zip(*importances1)]
    feature_importance1 = pd.DataFrame({"feature": list(columns), "importance": importances1})
    feature_importance1["importance"] = round(feature_importance1["importance"] * 100, 2)

    columns = X_norewards.columns
    importances2 = [sum(x) / len(x) for x in zip(*importances2)]
    feature_importance2 = pd.DataFrame({"feature": list(columns), "importance": importances2})
    feature_importance2["importance"] = round(feature_importance2["importance"] * 100, 2)

    return predictions, feature_importance1, feature_importance2


# load data
X_rewards = pd.read_csv(f"data\\data_IRL_with_rewards.csv", index_col=0)
X_val = pd.read_csv(f"data\\data_IRL_norewards.csv", index_col=0)
y_og = pd.read_csv(f"data\\label_IRL.csv", index_col=0)
y = list(y_og["bot"])

# Number of random trials
NUM_TRIALS = 5

# xgboost
param_grid = {"n_estimators" : [10, 50, 100, 150],
              "learning_rate": [0.03, 0.04],
              "booster": ["gbtree", "gblinear", "dart"]}

# estimator
estimator = XGBClassifier()
acc_xgb, roc_xgb, f1_xgb, spec_xgb, sens_xgb = nested_cross_val(NUM_TRIALS,
                                                                estimator,
                                                                param_grid,
                                                                X_rewards,
                                                                y)
acc_val, roc_val, f1_val, spec_val, sens_val = nested_cross_val(NUM_TRIALS,
                                                                estimator,
                                                                param_grid,
                                                                X_val,
                                                                y)
df_predictions, fi_rewards, fi_norewards = get_predictions_and_feature_importance(estimator,
                                                                               estimator,
                                                                               param_grid,
                                                                               X_rewards,
                                                                               y_og,
                                                                               X_val)

print(f'The feature importance for XGBoost with all features is: {fi_rewards}.')
print(f'The feature importance for XGBoost without rewards feature is: {fi_norewards}.')

p_value = delong_roc_test(df_predictions["ground_truth"], df_predictions["prediction_one"], df_predictions["prediction_two"])
print(f'The p-value for the Mann-Whitney U test for the two models is: {math.pow(10, p_value)}')
