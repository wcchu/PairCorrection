import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

MODEL_DIR = 'model_dir_py/'

# import data
data = pd.read_csv(
    'input.csv', dtype={
        'item1': str,
        'value1': float,
        'item2': str,
        'value2': float
    })
print('data size: {}'.format(len(data)))

x = data[['item1', 'value1', 'item2', 'value2']]
y = data['value2'] - data['value1']

# split training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)

# count number of unique item values
list_items = x_train['item1'].append(
    x_train['item2']).drop_duplicates().tolist()
n_items = len(list_items)
print("number of unique items: {}".format(n_items))

# columns
item1_col = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
        'item1', vocabulary_list=list_items))
value1_col = tf.feature_column.numeric_column('value1')
item2_col = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
        'item2', vocabulary_list=list_items))
value2_col = tf.feature_column.numeric_column('value2')


# build input function for training
input_fn_train = tf.estimator.inputs.pandas_input_fn(
    x=x_train,
    y=y_train,
    batch_size=100,
    num_epochs=3,
    shuffle=True
)


# build model function
def model_fn(features, labels, mode, params):
    input_layer = {
        'item1': tf.feature_column.input_layer(
            features={'item1': features['item1']},
            feature_columns=params['feature_columns'][0]),
        'value1': tf.feature_column.input_layer(
            features={'value1': features['value1']},
            feature_columns=params['feature_columns'][1]),
        'item2': tf.feature_column.input_layer(
            features={'item2': features['item2']},
            feature_columns=params['feature_columns'][2]),
        'value2': tf.feature_column.input_layer(
            features={'value2': features['value2']},
            feature_columns=params['feature_columns'][3])
    }

    # Build input layer for item1-value1 side:
    # Concatenate the 0th-order part (with item1 in one-hot vector) and
    # the 1st-order part (with value1 times item1 in one-hot vector)
    input_1 = tf.concat(
        (
            input_layer['item1'],
            tf.multiply(
                tf.concat([input_layer['value1']] * n_items, axis=1),
                input_layer['item1']
            )
        ),
        axis=1
    )

    input_2 = tf.concat(
        (
            input_layer['item2'],
            tf.multiply(
                tf.concat([input_layer['value2']] * n_items, axis=1),
                input_layer['item2']
            )
        ),
        axis=1
    )

    # force symmetry: when item1 and item2 swap, the response is the
    # same amplitude but changes sign
    subtracted = tf.subtract(input_1, input_2)

    out = tf.layers.dense(
        inputs=subtracted,
        units=1,
        use_bias=False,
        kernel_initializer=None)

    # define head
    my_head = tf.contrib.estimator.regression_head(
        label_dimension=1,
        loss_fn=None  # custom loss, default: mean_squared_error
    )

    return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        optimizer=tf.train.FtrlOptimizer(params['step_size']),
        logits=out,
    )


# custom estimator
regressor = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=MODEL_DIR,
    params={'feature_columns': [item1_col, value1_col, item2_col, value2_col],
            'step_size': 0.1}
)

# train estimator
regressor.train(
    input_fn=input_fn_train,
    steps=None
)


# evaluation input function
input_fn_eval = tf.estimator.inputs.pandas_input_fn(
    x=x_test,
    y=y_test,
    batch_size=1,
    num_epochs=1,
    shuffle=False
)


# evaluate
eval_result = regressor.evaluate(input_fn=input_fn_eval)
print(eval_result)


# extract trained variables in the model
checkpoint = tf.train.get_checkpoint_state(checkpoint_dir=MODEL_DIR)
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(
        checkpoint.model_checkpoint_path + '.meta')
    saver.restore(sess, checkpoint.model_checkpoint_path)
    tvs = sess.run(tf.trainable_variables())


# construct and export coefficient table
coeffs = np.reshape(tvs, [2, 704])
item_coeffs = pd.DataFrame(
    {'item': list_items, 'c': coeffs[0, :], 'd': coeffs[1, :]})
item_coeffs.to_csv('item_coeffs.csv', sep=',', index=False)
