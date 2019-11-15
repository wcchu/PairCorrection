# -*- coding: utf-8 -*-
"""
Build PairShift model with TensorFlow 2.0
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATA_FILE = 'input.csv'
EPOCHS = 5


def import_data(filename):
    data = pd.read_csv(filename,
                       dtype={
                           'item1': str,
                           'item2': str,
                           'dif': float
                       })
    print("number of data points: {}".format(len(data)))

    # split training and evaluation data
    d_train, d_eval = train_test_split(data, test_size=0.3)

    # count items
    items = d_train['item1'].append(d_train['item2']).unique().tolist()
    n_items = len(items)
    print("number of unique items: {}".format(n_items))

    return n_items, items, d_train, d_eval


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('dif')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


# define model
def create_pair_model(columns):
    '''
    Define pair model
    '''
    model = PairModel(columns)
    model.compile(optimizer=tf.keras.optimizers.Ftrl(0.2),
                  loss='mse',
                  metrics=['mae', 'mse'],
                  run_eagerly=False)
    return model


# PairModel is used to train variables using the paired examples
class PairModel(tf.keras.Model):
    '''
    Train variables using the paired examples

    Tutorial on basic structure:
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model
    '''

    def __init__(self, columns):
        super(PairModel, self).__init__()
        self.input1 = tf.keras.layers.DenseFeatures(columns['item1'])
        self.input2 = tf.keras.layers.DenseFeatures(columns['item2'])
        self.subtracted = tf.keras.layers.Subtract(name='sub')
        self.dense = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, inputs):
        # input1 and input2 take "item1" and "item2" respectively from
        # inputs
        hot1 = self.input1(inputs)
        hot2 = self.input2(inputs)
        # subtract the onehot of item2 from onehot of item1
        sub = self.subtracted([hot1, hot2])
        # simple linear regression
        out = self.dense(sub)
        return out


class PredModel(tf.Module):
    '''Predict each item's value'''

    def __init__(self, item_coefficients):
        '''
        Create a lookup table mapping from items to coefficients
        '''
        self.Table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(item_coefficients['item'], tf.string),
                values=tf.constant(item_coefficients['coefficient'],
                                   tf.float32)),
            default_value=0.0)

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, item):
        coeff = self.Table.lookup(item)
        return coeff


def run():
    """run"""

    # import data
    n_items, items, d_train, d_eval = import_data(DATA_FILE)

    # build datasets
    ds_train = df_to_dataset(d_train)
    ds_eval = df_to_dataset(d_eval, shuffle=False)

    # columns
    columns = {
        'item1':
            tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                    'item1', vocabulary_list=items)),
        'item2':
            tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                    'item2', vocabulary_list=items))
    }

    # create model
    pair_model = create_pair_model(columns)

    # train model
    pair_model.fit(ds_train, validation_data=ds_eval, epochs=EPOCHS)

    # print summary
    pair_model.summary()

    # directly get trained variables
    variables = tf.reshape(pair_model.trainable_variables,
                           shape=[n_items]).numpy()
    item_coefficients = pd.DataFrame({'item': items, 'coefficient': variables})
    item_coefficients.to_csv('item_coefficients.csv', index=False)

    # predict for each input item
    inputs = {
        'item1': np.array(items),
        'item2': np.empty(len(items), dtype=str)
    }
    predictions = pair_model.predict(inputs)
    print(predictions)

    # create prediction model
    pred_model = PredModel(item_coefficients)
    tf.saved_model.save(pred_model, export_dir='saved_pred_model')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.WARNING)
    run()
