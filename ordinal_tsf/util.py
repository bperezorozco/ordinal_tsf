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


def frame_generator_list(ts_list, frame_length, get_inputs, get_outputs, val_p, batch_size=256):
    train_list = []
    train_perm_list = []
    val_list = []
    val_perm_list = []

    total_tr_steps = 0
    total_val_steps = 0

    def single_fr_gen(ts, perm, frame_length):
        while True:
            np.random.shuffle(perm)
            for s in perm:
                yield ts[s:s+frame_length]

    for ts in ts_list:
        ts_length = ts.shape[0]
        ts_effective_length = ts_length - frame_length
        val_n = ts_effective_length - int(val_p * ts_effective_length)

        if int(val_p * ts_effective_length) < frame_length:
            continue

        train_list += [ts[:val_n]]
        val_list += [ts[val_n:]]

        train_perm_list += [np.random.permutation(ts[:val_n].shape[0] - frame_length)]
        val_perm_list +=  [np.random.permutation(ts[val_n:].shape[0] - frame_length)]

        total_tr_steps += train_perm_list[-1].size // batch_size
        total_val_steps += val_perm_list[-1].size // batch_size

    def fr_gen(ts_list, perm_list, frame_length, get_inputs, get_outputs, batch_size):
        while True:
            n_remaining = len(ts_list)
            # build individual generators
            ts_gens = [single_fr_gen(ts, perm, frame_length) for ts, perm in zip(ts_list, perm_list)]

            while n_remaining > 0:
                current_frames = []
                current_batch_size = 0
                insufficient_frames_left = False

                while current_batch_size < batch_size:
                    i_ts = np.random.choice(n_remaining)
                    frame = next(ts_gens[i_ts], None)

                    if frame is None:
                        ts_gens.pop(i_ts)
                        n_remaining -= 1

                        if n_remaining == 0:
                            insufficient_frames_left = True
                            break

                    else:
                        current_frames += [frame]
                        current_batch_size += 1

                if insufficient_frames_left: continue

                current_frames = np.stack(current_frames, axis=0)
                yield get_inputs(current_frames), get_outputs(current_frames)

    return fr_gen(train_list, train_perm_list, frame_length, get_inputs, get_outputs, batch_size), \
        fr_gen(val_list, val_perm_list, frame_length, get_inputs, get_outputs, batch_size), \
        total_tr_steps, total_val_steps


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








