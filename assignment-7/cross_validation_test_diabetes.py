
# Train/Test Split and Cross Validation in Python
# https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

# Classifier comparison
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

# Mark Needham
# scikit-learn - Random forests - Feature Importance
# http://www.markhneedham.com/blog/2017/06/16/scikit-learn-random-forests-feature-importance/

# Cross Validation and Model Selection
# http://pythonforengineers.com/cross-validation-and-model-selection/

# Improve Your Model Performance using Cross Validation (in Python and R)
# https://www.analyticsvidhya.com/blog/2015/11/improve-model-performance-cross-validation-in-python-r/

import os
import sys
import configdiabetes as config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import cm
import itertools
from tabulate import tabulate
from enum import Enum

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MinMaxScaler, MaxAbsScaler
from sklearn.neural_network import MLPClassifier    
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.externals import joblib 
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV
from xgboost import plot_importance

class Model(Enum):
    MLP_CLASSIFIER = 1
    LOGISTIC_REGRESSION = 2
    RANDOMFOREST_CLASSIFIER = 3
    ADA_BOOST_CLASSIFIER = 4
    GRADIENT_BOOST = 5
    DECISIONTREE_CLASSIFIER = 6
    SVM = 7
    GAUSSIAN_CLASSIFIER = 8
    GAUSSIANB_CLASSIFIER = 9
    KNEAREST_NEIGHBORS = 10
    XGBOOST_CLASSIFIER = 11


# #Choose best parameters for randomforest
def best_params(train_x, train_y):
    rfc = RandomForestClassifier()
    param_grid = { 
        'n_estimators': [50, 400],
        'max_features': ['auto', 'sqrt', 'log2']
    }    
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    CV_rfc.fit(train_x, train_y)
    return CV_rfc.best_params_

def format_decimal_number(decimal_number, decimal_digits=None):
    """
    format a decimal number
    :param decimal_number:
    :param decimal_digits:
    """
    try:
        if decimal_digits is None:
            format_value = float("{0:.2f}".format(decimal_number))
        else:
            if decimal_digits == 1:
                format_value = float("{0:.1f}".format(decimal_number))
            elif decimal_digits == 2:
                format_value = float("{0:.2f}".format(decimal_number))
            elif decimal_digits == 3:
                format_value = float("{0:.3f}".format(decimal_number))
            elif decimal_digits == 4:
                format_value = float("{0:.4f}".format(decimal_number))
            elif decimal_digits == 5:
                format_value = float("{0:.5f}".format(decimal_number))
            else:
                format_value = float("{0:.2f}".format(decimal_number))
    except Exception as ex:
        print( "An error occurred: {}".format(str(ex)))   
    return format_value

def plot_confusion_matrix(confusion_matrix, y_class, plot_title, plot_y_label, plot_x_label, normalize=False):
    """
    plot the confusion matrix.
    :param confusion_matrix: confusion matrix value
    :param y_class: target unique class name
    :param plot_title: plot title
    :param plot_y_label: plot y label
    :param plot_x_label: plot x label
    :param normalize: default to false
    :return: None
    """

    # If plotting is turned off, then return without generating any output
    if (not config.GENERATE_PLOTS):
        return

    try:
        plt.figure()
        plt.imshow(confusion_matrix, interpolation='nearest', cmap="Blues")
        plt.title(plot_title)
        plt.colorbar()
        tick_marks = np.arange(len(y_class))
        plt.xticks(tick_marks, y_class, rotation=45)
        plt.yticks(tick_marks, y_class)
#         if normalize:
#             confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
#             print("Normalized Confusion Matrix")
#         else:
#             print('Confusion Matrix without Normalization')    
        thresh = confusion_matrix.max() / 2.
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, confusion_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
        plt.ylabel(plot_y_label)
        plt.xlabel(plot_x_label)
        plt.tight_layout()    
        plt.show()
    except Exception as ex:
        print( "An error occurred: {}".format(str(ex)))   


def run_cross_validation_score(model, X, y, cv_folds_list):
    """
    calculate scores mean and std
    :param model: 
    :param X:
    :param y:
    :param cv_folds_list:
    """
    try:
        for cv_fold in cv_folds_list:
            scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv_fold)    
            scores_mean = format_decimal_number(scores.mean(), 2) * 100
            scores_std = format_decimal_number(scores.std() * 2, 4) * 100       
            print("Cross-Validaiton Fold: {}".format(cv_fold), file=output_report_file)
            print("Cross-Validaiton Score: {} % (+/- {} %)".format(scores_mean, scores_std), file=output_report_file)
            print(file=output_report_file)
    except Exception as ex:
        print( "An error occurred: {}".format(str(ex)))   


def run_model(classifier_model, model_type:Model, feature_name, has_feature_importances, X_train, y_train, X_test, y_test, y_unique_class, cv_folds_list):
    print("Running model: ", model_type.name, file=output_report_file)
    print(file=output_report_file)

    run_cross_validation_score(classifier_model, X_train, y_train, cv_folds_list)

    #     7. TRAIN THE MODEL WITH TRAIN DATA 
    classifier_model.fit(X_train, y_train)
    
#     7.1 FEATURE IMPORTANCES -------------------------------------------------------------------------
#         RANDOM FOREST / ADA BOOST CLASSIFIER / GRADIENT BOOSTING CLASSIFIER / DECISION TREE CLASSIFIER"
    if (has_feature_importances):
        print("Feature Importances for " + model_type.name)
        print(sorted(zip(feature_name, classifier_model.feature_importances_), key=lambda x: x[1] * -1))
        print()
#       --------------------------------------------------------------------------------------------------------------------

#       7.2 FOR TABULATE LIBRARY ONLY-------------------------------------------------------------------
        print("Feature Importances for " + model_type.name, file=output_report_file)
        headers = ["name", "score"]
        values = sorted(zip(feature_name, classifier_model.feature_importances_), key=lambda x: x[1] * -1)       
        print(tabulate(values, headers, tablefmt="rst"), file=output_report_file)
        print(file=output_report_file)

        # Plot feature_importances_ if present
        if (config.GENERATE_PLOTS):
            if (model_type is Model.XGBOOST_CLASSIFIER):
                plot_importance(classifier_model)
                plt.show()
            else:
                plt.title("Feature Importance for " + model_type.name, loc="center")
                plt.bar(range(len(classifier_model.feature_importances_)), classifier_model.feature_importances_)
                plt.show()

#     #     ---------------------------------------------------------------------------------------------------------------------
    
#     8. GET TARGET PREDICTED VALUES
    y_predicted = classifier_model.predict(X_test)      
      
#     9. MODEL EVALUATION FOR DATA CLASSIFICATION
#     ACCURACY SCORE
    accuracy_score_value = accuracy_score(y_test, y_predicted) * 100
    accuracy_score_value = float("{0:.2f}".format(accuracy_score_value))    
    print("Accuracy Score for " + model_type.name + ": {} %".format(accuracy_score_value), file=output_report_file)
    print(file=output_report_file)
    
#     CONFUSION MATRIX
    confusion_matrix_result = confusion_matrix(y_test, y_predicted)
    print("Confusion Matrix:", file=output_report_file)
    print(confusion_matrix_result, file=output_report_file)
    print(file=output_report_file)
    
#     CLASSIFICATION REPORT
    classification_report_result = classification_report(y_test,y_predicted)
    print("Classification Report:", file=output_report_file)    
    print(classification_report_result, file=output_report_file)
    print(file=output_report_file)  
    
#     SHOW CONFUSION MATRIX PLOT
    plot_confusion_matrix(confusion_matrix_result, 
                          y_class=y_unique_class, 
                          plot_title="Confusion Matrix for " + model_type.name + " - " + config.INPUT_DATA + " Data", 
                          plot_y_label="Test " + config.INPUT_DATA + " Class", 
                          plot_x_label="Predicted " + config.INPUT_DATA + " Class")      
        
#     10. SAVE MODEL TO A .PKL FILE    
    joblib.dump(classifier_model, "classifier_model_file.pkl")
    
#     11. LOAD MODEL FROM .PKL FILE
    mlp_classifier_loaded = joblib.load('classifier_model_file.pkl')
 
#     12. PREDICT DATA SET USING LOADED MODEL
    mlp_classifier_loaded.predict(X_test)

# HOW TO IMPROVE THE MODEL?

#     1. TRY OTHER CLASSIFICATION MODEL FAMILIES (LOGISTIC REGRESSION, RANDOM FOREST, SUPPORT VECTOR MACHINES, ETC.).
#     2. COLLECT MORE DATA IF IT'S CHEAP TO DO SO.
#     3. ENGINEER SMARTER FEATURES AFTER SPENDING MORE TIME ON EXPLORATORY ANALYSIS.
#     4. SPEAK TO A DOMAIN EXPERT TO GET MORE CONTEXT INFORMATION



####################################################################################################
# main
####################################################################################################
def main():   

# COLUMN NAMES
#   sepal-length
#   sepal-width
#   petal-length
#   petal-width
#   class: 0, 1, 2

    # Get default project directory path
    project_directory_path = os.path.dirname(sys.argv[0])  
    input_file_path = os.path.join(project_directory_path, config.INPUT_FILE_PATH)
    output_report_path = os.path.join(project_directory_path, config.OUTPUT_REPORT_PATH)

#   Open output report file for append
    global output_report_file
    output_report_file = open(output_report_path, 'a')
    

#     1. LOAD DATA INTO PANDAS DATAFRAME
    df_data = pd.read_csv(filepath_or_buffer=input_file_path)
    print(df_data)
    print()
#     
#     DATA PREPROCESSING
#    df_data['preg_count'] = df_data['preg_count'].map( lambda x : df_data.preg_count.median() if x == 0 else x)
#    df_data['glucose_concentration'] = df_data['glucose_concentration'].map( lambda x : df_data.glucose_concentration.median() if x == 0 else x)
#    df_data['blood_pressure'] = df_data['blood_pressure'].map( lambda x : df_data.blood_pressure.median() if x == 0 else x)
#    df_data['skin_thickness'] = df_data['skin_thickness'].map( lambda x : df_data.skin_thickness.median() if x == 0 else x)
#    df_data['serum_insulin'] = df_data['serum_insulin'].map( lambda x : df_data.serum_insulin.median() if x == 0 else x)
#    df_data['bmi'] = df_data['bmi'].map( lambda x : df_data.bmi.median() if x == 0 else x)
    
#         this code needs to be test it!
#     df_data = df_data.replace(0, np.nan)
#     df_data.fillna(value=df_data.median(), inplace=True)
#     df_data = df_data.fillna(df_data.median())

#     SHOW df_data AFTER DATA PREPROCESSING
#     print(df_data)
#     print()
    
#     PRINT DATAFRAME INFORMATION
#     df_data.info()    
#     print()
#     2. DEFINE THE FEATURES 
    X = df_data.drop(labels=config.OUTCOME, axis=1)
    feature_name = X.columns.values 
    
#     3. DEFINE THE TARGET
    y = df_data[config.OUTCOME]
    y_unique_class = list(y.unique())
    print(y)
        
#     4. GET TRAIN AND TEST DATA 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    
#     5. SCALE THE DATA - STANDARDIZE FEATURES BY REMOVING THE MEAN AND SCALING TO UNIT VARIANCE
    scaler = StandardScaler()    
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
#     result = 76.62 %
    
#     robust_scaler = RobustScaler(quantile_range=(25, 75))  
#     robust_scaler.fit_transform(X_train)
#     X_train = robust_scaler.transform(X_train)
#     X_test = robust_scaler.transform(X_test)
#     result = 75.76 %
    
#     normalizer_scaler = Normalizer()
#     normalizer_scaler.fit_transform(X_train)
#     X_train = normalizer_scaler.transform(X_train)
#     X_test = normalizer_scaler.transform(X_test)
#     result = 63.2 %
    
#     min_max_scaler = MinMaxScaler()
#     min_max_scaler.fit_transform(X_train)
#     X_train = min_max_scaler.transform(X_train)
#     X_test = min_max_scaler.transform(X_test)
#     result = 76.62 %

#     max_abs_scaler = MaxAbsScaler()
#     max_abs_scaler.fit_transform(X_train)
#     X_train = max_abs_scaler.transform(X_train)
#     X_test = max_abs_scaler.transform(X_test)
#     result = 75.32%

#     ---------------------------------------------------------------------------------------------    
#     6.1 CREATE MULTI-LAYER PERCEPTRON CLASSIFIER MODEL
    if (config.RUN_MLP_CLASSIFIER):
        model_type = Model.MLP_CLASSIFIER
        print("MULTI-LAYER PERCEPTRON CLASSIFIER", file=output_report_file)
        classifier_model = MLPClassifier(activation="identity", hidden_layer_sizes=(100, 100, 100), max_iter=1000, random_state=1)
        cv_folds_list = [5, 10, 15, 20]    
        run_model(classifier_model, model_type, feature_name, False, X_train, y_train, X_test, y_test, y_unique_class, cv_folds_list)
    
#     6.2 BUILDING A LOGISTIC REGRESSION IN PYTHON, STEP BY STEP
#     https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
    if (config.RUN_LOGISTIC_REGRESSION):
        print("LOGISTIC REGRESSION CLASSIFIER", file=output_report_file)
        model_type = Model.LOGISTIC_REGRESSION
        classifier_model = LogisticRegression(random_state=1, solver="liblinear", max_iter=1000)    
        cv_folds_list = [5, 10, 15, 20]    
        run_model(classifier_model, model_type, feature_name, False, X_train, y_train, X_test, y_test, y_unique_class, cv_folds_list)
        
#     6.3 RANDOM FOREST CLASSIFIER
    if (config.RUN_RANDOMFOREST_CLASSIFIER):
        print("RANDOM FOREST CLASSIFIER", file=output_report_file)
        best_rfc_params = best_params(X_train, y_train)
        print ("Best params:")
        print(best_rfc_params)
        print()

        model_type = Model.RANDOMFOREST_CLASSIFIER
        #classifier_model = RandomForestClassifier(criterion="gini", max_depth=5, n_estimators=10, max_features=1, n_jobs=2, random_state=1)
        #classifier_model = RandomForestClassifier(criterion="gini", max_depth=5, n_estimators=10, max_features=2, n_jobs=2, random_state=1)
        #classifier_model = RandomForestClassifier(criterion="entropy", max_depth=None, n_estimators=10, max_features="auto", n_jobs=-1, random_state=1)
        classifier_model = RandomForestClassifier(criterion="gini", max_depth=None, n_estimators=10, max_features="auto", n_jobs=-1, random_state=1)
        # 97.78% accuracy score
        cv_folds_list = [5, 10, 15, 20]    
        run_model(classifier_model, model_type, feature_name, True, X_train, y_train, X_test, y_test, y_unique_class, cv_folds_list)
    
#     6.4 ADA BOOST CLASSIFIER
    if (config.RUN_ADA_BOOST):
        print("ADA BOOST CLASSIFIER", file=output_report_file)
        model_type = Model.ADA_BOOST_CLASSIFIER
        classifier_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME", n_estimators=200, random_state=1)
        cv_folds_list = [5, 10, 15, 20]    
        run_model(classifier_model, model_type, feature_name, True, X_train, y_train, X_test, y_test, y_unique_class, cv_folds_list)
    
#     6.5 GRADIENT BOOSTING CLASSIFIER
    if (config.RUN_GRADIENT_BOOST):
        print("GRADIENT BOOSTING CLASSIFIER", file=output_report_file)
        model_type = Model.GRADIENT_BOOST
        classifier_model = GradientBoostingClassifier(n_estimators=1000, criterion="friedman_mse", max_leaf_nodes=4, random_state=1)
        cv_folds_list = [5, 10, 15, 20]    
        run_model(classifier_model, model_type, feature_name, True, X_train, y_train, X_test, y_test, y_unique_class, cv_folds_list)
    
#     6.6 DECISION TREE CLASSIFIER
    if (config.RUN_DECISIONTREE_CLASSIFIER):
        print("DECISION TREE CLASSIFIER", file=output_report_file)
        model_type = Model.DECISIONTREE_CLASSIFIER
        classifier_model = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=3, min_samples_leaf=5, random_state=1)        
        cv_folds_list = [5, 10, 15, 20]    
        run_model(classifier_model, model_type, feature_name, True, X_train, y_train, X_test, y_test, y_unique_class, cv_folds_list)
    
#     6.7 SUPPORT VECTOR MACHINES
    if (config.RUN_SVM):
        print("SUPPORT VECTOR MACHINES", file=output_report_file)   
        model_type = Model.SVM 
        classifier_model = SVC(C=1, kernel="linear", gamma=1, random_state=1)       
        cv_folds_list = [5, 10, 15, 20]    
        run_model(classifier_model, model_type, feature_name, False, X_train, y_train, X_test, y_test, y_unique_class, cv_folds_list)

#    6.8 GAUSSIAN PROCESS CLASSIFIER
    if (config.RUN_GAUSSIAN_CLASSIFIER):
        print("GAUSSIAN PROCESS CLASSIFIER", file=output_report_file) 
        model_type = Model.GAUSSIAN_CLASSIFIER  
        classifier_model = GaussianProcessClassifier(kernel=1.0 * RBF(1.0), optimizer="fmin_l_bfgs_b", random_state=1)       
        cv_folds_list = [5, 10, 15, 20]    
        run_model(classifier_model, model_type, feature_name, False, X_train, y_train, X_test, y_test, y_unique_class, cv_folds_list)
#     result = 79.0 % (+/- 3.53 %) - very good!

#     6.9 GAUSSIAN NAIVE BAYES (GAUSSIANNB)
    if (config.RUN_GAUSSIANB_CLASSIFIER):
        print("GAUSSIAN NAIVE BAYES (GAUSSIANNB)", file=output_report_file) 
        model_type = Model.GAUSSIANB_CLASSIFIER  
        classifier_model = GaussianNB()
        cv_folds_list = [5, 10, 15, 20]    
        run_model(classifier_model, model_type, feature_name, False, X_train, y_train, X_test, y_test, y_unique_class, cv_folds_list)

#     6.10 K-NEAREST NEIGHBORS CLASSIFIER
    if (config.RUN_KNEAREST_NEIGHBORS):
        print("K-NEAREST NEIGHBORS CLASSIFIER", file=output_report_file)
        model_type = Model.KNEAREST_NEIGHBORS
        classifier_model = KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="auto")
        cv_folds_list = [5, 10, 15, 20]    
        run_model(classifier_model, model_type, feature_name, False, X_train, y_train, X_test, y_test, y_unique_class, cv_folds_list)
    
#     6.11 XGBOOST CLASSIFIER
    if (config.RUN_XGBOOST_CLASSIFIER):
        print("XGBOOST CLASSIFIER", file=output_report_file)
        model_type = Model.XGBOOST_CLASSIFIER
#       need to apply GridSearchCV() to determine best hyperparameters
        classifier_model = XGBClassifier()
        cv_folds_list = [5, 10, 15, 20]    
        #run_cross_validation_score(classifier_model, X_train, y_train, cv_folds_list)
        #print(file=output_report_file)     

        XGBClassifier(base_score=0.5, colsample_bytree=1, gamma=0, learning_rate=0.1,
        max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
        n_estimators=100, nthread=-1, objective='multi:softprob', seed=0,
        silent=True, subsample=1)

        run_model(classifier_model, model_type, feature_name, True, X_train, y_train, X_test, y_test, y_unique_class, cv_folds_list)


#     --------------------------------------------------------------------------------------------------------------------


        
if __name__ == '__main__':
    main()
