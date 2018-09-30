# Import libraries
import matplotlib.pyplot as plt
import itertools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.metrics import precision_score, f1_score
from keras.callbacks import Callback


# This class selects the desired attributes and drops the rest, and converts the DataFrame to a Numpy array.
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class LabelBinarizerForPipeline(LabelBinarizer):

    def fit(self, X, y=None):
        return super(LabelBinarizerForPipeline, self).fit(X)

    def transform(self, X):
        return super(LabelBinarizerForPipeline, self).transform(X)

    def fit_transform(self, X, Y=None):
        result = [super(LabelBinarizerForPipeline, self).fit(col).transform(col) for col in X.T]
        return np.column_stack(result)


# This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # plt.figure(figsize=(7, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# This class computes the required metrics after each training epoch.
# It is passed as a callback to the Keras model fit() function.
class Metrics(Callback):

    def __init__(self, train_data):

        super(Metrics, self).__init__()
        self.train_data = train_data

    def on_train_begin(self, logs={}):

        self.train_f1s_macro = []
        self.train_f1s_micro = []
        self.train_precisions_macro = []
        self.train_precisions_micro = []

        self.val_f1s_macro = []
        self.val_f1s_micro = []
        self.val_precisions_macro = []
        self.val_precisions_micro = []

    def on_epoch_end(self, epoch, logs={}):
        x_train, y_train = self.train_data
        train_predict = (np.asarray(self.model.predict(x_train))).round()

        _train_f1_macro = f1_score(y_train, train_predict, average='macro')
        _train_f1_micro = f1_score(y_train, train_predict, average='micro')
        _train_precision_macro = precision_score(y_train, train_predict, average='macro')
        _train_precision_micro = precision_score(y_train, train_predict, average='micro')

        self.train_f1s_macro.append(_train_f1_macro)
        self.train_f1s_micro.append(_train_f1_micro)
        self.train_precisions_macro.append(_train_precision_macro)
        self.train_precisions_micro.append(_train_precision_micro)

        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]

        _val_f1_macro = f1_score(val_targ, val_predict, average='macro')
        _val_f1_micro = f1_score(val_targ, val_predict, average='micro')
        _val_precision_macro = precision_score(val_targ, val_predict, average='macro')
        _val_precision_micro = precision_score(val_targ, val_predict, average='micro')

        self.val_f1s_macro.append(_val_f1_macro)
        self.val_f1s_micro.append(_val_f1_micro)
        self.val_precisions_macro.append(_val_precision_macro)
        self.val_precisions_micro.append(_val_precision_micro)

        return None
