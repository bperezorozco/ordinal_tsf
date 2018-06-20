import keras.backend as K
import numpy as np
import itertools


def cartesian(all_lists):
    return itertools.product(*all_lists)


def assert_sum_one(x_list): return np.isclose(sum(x_list), 1.)


def all_satisfy(x_list, condition): return all([condition(x) for x in x_list])


def assert_keys_in_dct(kw, keys): return all([kw.has_key(k) for k in keys])


def ore_mse(y_true, y_pred):
    return K.mean(K.sqrt(K.sum((y_true.argmax(axis=-1) - y_pred.argmax(axis=-1)) ** 2, axis=-1)), axis=-1)


def ore_mse_tf(y_true, y_pred):
    sq_errors = K.sum((K.argmax(y_true, axis=-1) - K.argmax(y_pred, axis=-1)) ** 2, axis=-1)
    return K.mean(K.sqrt(K.tf.cast(sq_errors, dtype=K.tf.float32)), axis=-1)


def is_univariate(ts):
    if ts.ndim == 1: return True
    if ts.ndim == 2 and ts.shape[1] == 1: return True

    return False


def frame_ts(ts, frame_length, hop=1):
    ts_length = ts.shape[0]
    assert frame_length < ts_length, 'Choose timesteps < {}'.format(ts_length)

    s_idx = np.arange(0, ts_length - frame_length, hop)
    return np.array([ts[s:s + frame_length] for s in s_idx])
