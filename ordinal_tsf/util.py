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


def frame_generator(ts, frame_length, get_inputs, get_outputs, val_p, batch_size=256):
    ts_length = ts.shape[0]
    assert frame_length < ts_length, 'Choose timesteps < {}'.format(ts_length)

    this_perm = np.random.permutation(ts_length - frame_length)
    val_index = int(val_p * this_perm.shape[0])
    tr_s_idx = this_perm[:val_index]
    val_s_idx = this_perm[val_index:]
    tr_steps = tr_s_idx.size // batch_size
    val_steps = val_s_idx.size // batch_size

    def fr_gen(ts, s_idx, frame_length, get_inputs, get_outputs, batch_size):
        n_batches = s_idx.size // batch_size
        while True:
            np.random.shuffle(s_idx)

            for i in range(1, n_batches + 1):
                frames = np.array([ts[s:s + frame_length] for s in s_idx[(i - 1) * batch_size:i * batch_size]])
                yield get_inputs(frames), get_outputs(frames)

    return fr_gen(ts, tr_s_idx, frame_length, get_inputs, get_outputs, batch_size), \
           fr_gen(ts, val_s_idx, frame_length, get_inputs, get_outputs, batch_size),\
           tr_steps, val_steps








