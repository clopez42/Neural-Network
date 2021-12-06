from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import tensorflow_probability as tfp

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from numpy import sqrt

# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data = pd.read_csv(r"C:\Users\clope\OneDrive\Fall 2021 Semester\CS461_Artificial Intelligence\StudentsPerformance.csv")
CATEGORICAL_COLUMNS = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
SCORE_COLUMNS = ['math score', 'reading score', 'writing score']

df = pd.DataFrame(data, columns= ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course',
                                  'math score', 'reading score', 'writing score'])

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = df[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
for feature_name in SCORE_COLUMNS:
     feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.int32))
# print(feature_columns)

def input_fn(features, labels, training=True, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000)
    return dataset.batch(batch_size)

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[10,5], n_classes=3)

classifier.train(input_fn=lambda: input_fn(df, df, training=True), steps=5000)

X, y = df.values[0:4,:-1], df.values[5:7,:-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
n_features = X_train.shape[1]
print(n_features)

model = Sequential()
model.add(Dense(4, activation='sigmoid', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(3, activation='sigmoid', kernel_initializer='he_normal'))
model.add(Dense(1))

opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='mse')

history = model.fit(X, y, epochs=100, batch_size=32, verbose=0)

error = model.evaluate(X_test, y_test)
print('MSE: %.3f, RMSE: %.3f' %(error, sqrt(error)))

yhat = model.predict()
print('Predicted: %.3f' % yhat)

# model.summary()

plt.title('Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label= 'val')
plt.legend()
plt.show()



