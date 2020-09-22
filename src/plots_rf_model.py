import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, classification_report, roc_auc_score, make_scorer
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix #, plot_confusion_matrix
from plot_models import plot_learning_curves, plot_confusion_matrix
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


    # #feature importances
    feat_scores_array = model.feature_importances_
    feat_scores_df = pd.DataFrame({'Fraction of Samples Affected by Feature' : model.feature_importances_},
                            )
    feat_scores = feat_scores_df.sort_values(by='Fraction of Samples Affected by Feature', ascending=False)

    # #plot top 40 features
    fig, ax = plt.subplots()
    x_pos = np.arange(len(feat_scores[:40]))
    ax.barh(x_pos, feat_scores['Fraction of Samples Affected by Feature'][:40], align='center')
    plt.yticks(x_pos, feat_scores.index[:40])
    ax.set_ylabel('MFCC Coefficients')
    ax.set_xlabel('Fraction of Samples Affected')
    ax.set_title('MFCC Coefficients by Importance')
    plt.gca().invert_yaxis()
    plt.savefig('images/top40_rf_feature_importances.png', bbox_inches = "tight")
    plt.show()

    # #plot top 15 features
    fig, ax = plt.subplots()
    x_pos = np.arange(len(feat_scores[:15]))
    ax.barh(x_pos, feat_scores['Fraction of Samples Affected by Feature'][:15], align='center')
    plt.yticks(x_pos, feat_scores.index[:15])
    ax.set_ylabel('MFCC Coefficients')
    ax.set_xlabel('Fraction of Samples Affected')
    ax.set_title('Top 15 MFCC Coefficients by Importance')
    plt.gca().invert_yaxis()
    plt.savefig('images/top20_rf_feature_importances.png', bbox_inches = "tight")
    plt.show()

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    # Plot confusion matrix
    class_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet',
                    'organ', 'reed', 'string', 'synth_lead', 'vocal']
    plot_confusion_matrix(cm, classes=class_names,
                          title='Confusion Matrix', image_name='rf_model')
