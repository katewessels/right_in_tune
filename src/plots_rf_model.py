import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, classification_report, roc_auc_score, make_scorer
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import seaborn as sns
sns.set()
import itertools


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

def plot_confusion_matrix(cm, classes, model_name, title='Confusion Matrix', normalize=False, cmap=plt.cm.Blues):
    plt.figure()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.title(title, fontsize=18)
    plt.grid(False)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
                )
    plt.savefig(f'images/confusion_matrix_{model_name}.png', bbox_inches='tight')
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
    print(f'accuracy: {accuracy}')
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


    plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix', normalize=False, model_name='rf')
    plot_confusion_matrix(cm, classes=class_names, title='Normalized Confusion Matrix', normalize=True, model_name='rf_normalized')


    #PLOT MISCLASSIFICATION RATES
    #confusion matrix for final rf model
    # bass[[11866    25    33   231   476   166    78    59    24    21    56]
    # brass [    0  2540     2     1     3     0     0     5     8     0     2]
    # flute[   10    11  1696    10     1     5     0    20     3     1     6]
    # guitar[  218     8     7  5905   247   137    42    16    10    18    14]
    # keyboard[  258    21     6   219  9532   176    39    27    18    26    13]
    # mallet[  107    11     8    81   131  6383    23     7     8    23    25]
    # organ[   22     5     4     9    70    16  6677    12     5     9    11]
    # reed [   45    21    14     4     9     0     3  2765     4     2     8]
    # string [   20     2     1    18    10     2     2     1  3807     0     3]
    # synth-lead [   37     2     5     8    14    15     5     1     2   995     0]
    # vocal [   20     7     4     1     5     0     3     6     0     0  2007]]

    #misclassification rates
    x = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed',
        'string', 'synth-lead',  'vocal']
    y = [.089, .0081, .038, .11, .078, .062,
        .024, .038, .015, .0821, .022]
    #reorder indices, ascending
    classes = []
    values = []
    for idx in np.argsort(y):
        classes.append(x[idx])
        values.append(y[idx])
    #plot
    fig, ax = plt.subplots(figsize=(12,8))
    ax.barh(classes, values)
    ax.set_xlabel('Fraction Misclassified', fontsize=18)
    plt.yticks(fontsize=22)
    ax.set_title('Fraction of Misclassifications by Class', fontsize=26)
    plt.savefig('images/fraction_misclassified.png', bbox_inchs='tight')
    plt.show()