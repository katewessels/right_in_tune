import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, classification_report, roc_auc_score, make_scorer
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from plot_models import plot_confusion_matrix
import seaborn as sns
sns.set()


def save_pickle_files(file_, file_path):
    with open(file_path, 'wb') as f:
        # Write the model to a file.
        pickle.dump(file_, f)
    return None

def get_pickle_files(file_path):
    with open(file_path, 'rb') as f:
        file_ = pickle.load(f)
    return file_

def plot_top_features(model, num_features, xlabel, ylabel, title, ax):
    # feat_scores_array = model.feature_importances_
    feat_scores_df = pd.DataFrame({'Fraction of Samples Affected by Feature' : model.feature_importances_},                )
    feat_scores = feat_scores_df.sort_values(by='Fraction of Samples Affected by Feature', ascending=False)
    x_pos = np.arange(len(feat_scores[:num_features]))
    ax.barh(x_pos, feat_scores['Fraction of Samples Affected by Feature'][:num_features], align='center')
    plt.yticks(x_pos, feat_scores.index[:num_features])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.gca().invert_yaxis()
    plt.savefig(f'images/top{num_features}_rf_feature_importances.png', bbox_inches = "tight")
    plt.show()

if __name__ == "__main__":

    #get mfcc feature data
    X = get_pickle_files('pickles/X.pkl')
    y = get_pickle_files('pickles/y.pkl')
    featuresdf = get_pickle_files('pickles/featuresdf.pkl')

    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=127)

    #set up test and train sets first, which we already have
    sm = SMOTE(random_state=462)
    # print('Original dataset shape %s' % Counter(y_train))
    X_train, y_train = sm.fit_sample(X_train, y_train)
    # print('Resampled dataset shape %s' % Counter(y_train))


    #get model
    model = get_pickle_files('pickles/rf_model.pkl')

    #test predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    #metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    #classification report
    report = classification_report(y_test, y_pred)
    #training subset predict
    train_y_pred = model.predict(X_train)
    train_y_pred_proba = model.predict_proba(X_train)
    train_accuracy = accuracy_score(y_train, train_y_pred)
    train_recall = recall_score(y_train, train_y_pred, average='macro')
    train_precision = precision_score(y_train, train_y_pred, average='macro')
    train_auc = roc_auc_score(y_train, train_y_pred_proba, multi_class='ovr')



    #PLOT METRICS
    #plot the 40 MFCC Features
    fig, ax = plt.subplots()
    plot_top_features(model, 40, 'Fraction of Samples Affected', 'MFCC Coefficients', 'MFCC Coefficients by Importance', ax=ax)

    #plot the top 12 MFCC Features
    fig, ax = plt.subplots()
    plot_top_features(model, 12, 'Fraction of Samples Affected', 'MFCC Coefficients', 'Top 12 MFCC Features by Importance', ax=ax)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    # Plot confusion matrix
    class_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet',
                    'organ', 'reed', 'string', 'synth_lead', 'vocal']
    plot_confusion_matrix(cm, classes=class_names,
                          title='Confusion Matrix', image_name='rf_model')
