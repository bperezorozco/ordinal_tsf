from __future__ import division
import keras.backend as K
import numpy as np
import itertools


def gmm_marginal_pdf(x, channel_marginals, state_probs, dx=1.):
    pdf_mixtures = np.stack([gaussian.pdf(x) for gaussian in channel_marginals]).squeeze()
    return state_probs.dot(pdf_mixtures) * dx


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
    na_mask = np.isnan(ts)

    if na_mask.any():
        x_na = na_mask.any(axis=1)
        changepoints = np.logical_xor(x_na[1:], x_na[:-1])
        changepoints_idx = np.concatenate([[0], np.where(changepoints)[0] + 1])

        ts_segments = [ts[changepoints_idx[i]:changepoints_idx[i + 1]]
                       for i in range(changepoints_idx.shape[0] - 1)]
        ts_segments = [p for p in ts_segments if np.isnan(p).any(axis=1).all() == False and p.shape[0] >= frame_length]

        return np.concatenate([frame_ts(p, frame_length) for p in ts_segments], axis=0)

    s_idx = np.arange(0, ts_length - frame_length, hop)
    return np.array([ts[s:s + frame_length] for s in s_idx])


def to_contiguous(X):
    if X.ndim == 2:
        return X.reshape(-1, 1)

    n_frames = X.shape[0]

    return X.reshape(n_frames, -1)


def to_channels(X, n_channels):
    # this assumes X has [x1_1, x2_1, .. xnc_1, x1_2, x2_2, ... xn2_2 x1_3........... xnc_ts]
    return X.reshape(X.shape[0], -1, n_channels)


def frame_generator_multichannel(ts_list, frame_length, get_inputs, get_outputs, val_p, batch_size=256):
    ts_length = ts_list[0].shape[0]
    ts_effective_length = ts_length - frame_length
    val_n = ts_effective_length - int(val_p * ts_effective_length)

    train_list = [ts[:val_n] for ts in ts_list]
    val_list = [ts[val_n:] for ts in ts_list]

    train_perm = np.random.permutation(val_n - frame_length)
    val_perm = np.random.permutation(ts_length - val_n - frame_length)

    total_tr_steps = train_perm.size // batch_size
    total_val_steps = val_perm.size // batch_size

    def fr_gen(ts_list, s_idx, frame_length, get_inputs, get_outputs, batch_size):
        n_batches = s_idx.size // batch_size
        while True:
            np.random.shuffle(s_idx)
            for i in range(1, n_batches + 1):
                frames = [np.array([ts[s:s + frame_length] for s in s_idx[(i - 1) * batch_size:i * batch_size]])
                          for ts in ts_list]
                yield get_inputs(frames), get_outputs(frames)

    return fr_gen(train_list, train_perm, frame_length, get_inputs, get_outputs, batch_size), \
           fr_gen(val_list, val_perm, frame_length, get_inputs, get_outputs, batch_size), \
           total_tr_steps, total_val_steps

# Assumes all TS in ts_list have the same length

# This enables the creation of mixed frame generators (drawn from different time series in ts_list)
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

    last_frame_limit = ts_length - frame_length

    if ts.ndim == 2:
        na_mask = np.isnan(ts[:, :, np.newaxis]).any(axis=-1).any(axis=-1)
    else:
        na_mask = np.isnan(ts).any(axis=-1).any(axis=-1)

    not_vetted_idx = np.logical_not(np.array([na_mask[s:s+frame_length].any() for s in range(last_frame_limit)]))

    this_perm = np.arange(last_frame_limit)[not_vetted_idx]
    np.random.shuffle(this_perm)

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


def get_optimal_1dkde_bandwidth(t, min_val=1e-3):
    n = t.shape[0]

    return max(1.06 * t.std() * (n ** -0.2), min_val)


def nanmin(x, y, nan_is_greater=True):
    out = x < y
    out[np.isnan(x)] = not nan_is_greater
    out[np.isnan(y)] = nan_is_greater

    return out


def highlight_min(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_min = s == s.min()
    return ['background-color: lightblue' if v else '' for v in is_min]


def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_min = s == s.max()
    return ['background-color: lightblue' if v else '' for v in is_min]


def bold_min(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_min = s == s.min()
    return ['font-weight: bold' if v else '' for v in is_min]


def bold_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_min = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_min]


def get_continuous_runs(x):
    t = np.arange(x.shape[0])
    breakpoints = t[1:][(x[1:] - x[:-1]) != 0]

    if breakpoints.shape[0] == 0: return {x[0]: []}

    runs_by_cluster = {x[0]: [t[:breakpoints[0]]]}

    for i in range(breakpoints.shape[0] - 1):
        if x[breakpoints[i]] in runs_by_cluster:
            runs_by_cluster[x[breakpoints[i]]] += [t[breakpoints[i]:breakpoints[i + 1]]]
        else:
            runs_by_cluster[x[breakpoints[i]]] = [t[breakpoints[i]:breakpoints[i + 1]]]

    return runs_by_cluster


def transition_matrix(transitions):
    n = 1 + max(transitions)  # number of states

    M = [[0] * n for _ in range(n)]

    for (i, j) in zip(transitions, transitions[1:]):
        M[i][j] += 1

    # now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f / s for f in row]
    return np.array(M)


### DBA https://github.com/fpetitjean/DBA/blob/master/DBA.py
#
# __author__ ="Francois Petitjean"


'''
/*******************************************************************************
 * Copyright (C) 2018 Francois Petitjean
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
'''

import numpy as np
from functools import reduce


def performDBA(series, n_iterations=10):
    n_series = len(series)
    max_length = reduce(max, map(lambda x: x.shape[1], series))

    cost_mat = np.zeros((max_length, max_length))
    delta_mat = np.zeros((max_length, max_length))
    tmp_delta_mat = np.zeros((max_length, max_length))
    path_mat = np.zeros((max_length, max_length), dtype=np.int8)

    medoid_ind = approximate_medoid_index(series,cost_mat,delta_mat,tmp_delta_mat)
    center = series[medoid_ind]

    for i in range(0,n_iterations):
        center = DBA_update(center, series, cost_mat, path_mat, delta_mat,tmp_delta_mat)

    return center


def approximate_medoid_index(series,cost_mat,delta_mat,tmp_delta_mat):
    if len(series)<=50:
        indices = range(0,len(series))
    else:
        indices = np.random.choice(range(0,len(series)),50,replace=False)

    medoid_ind = -1
    best_ss = 1e20
    for index_candidate in indices:
        candidate = series[index_candidate]
        ss = sum_of_squares(candidate,series,cost_mat,delta_mat,tmp_delta_mat)
        if(medoid_ind==-1 or ss<best_ss):
            best_ss = ss
            medoid_ind = index_candidate
    return medoid_ind


def sum_of_squares(s,series,cost_mat,delta_mat,tmp_delta_mat):
    return sum(map(lambda t:squared_DTW(s,t,cost_mat,delta_mat,tmp_delta_mat),series))


def squared_DTW(s,t,cost_mat,delta_mat,tmp_delta_mat):
    s_len = len(s)
    t_len = len(t)
    length = len(s)
    fill_delta_mat_dtw(s, t, delta_mat,tmp_delta_mat)
    cost_mat[0, 0] = delta_mat[0, 0]
    for i in range(1, s_len):
        cost_mat[i, 0] = cost_mat[i-1, 0]+delta_mat[i, 0]

    for j in range(1, t_len):
        cost_mat[0, j] = cost_mat[0, j-1]+delta_mat[0, j]

    for i in range(1, s_len):
        for j in range(1, t_len):
            diag,left,top =cost_mat[i-1, j-1], cost_mat[i, j-1], cost_mat[i-1, j]
            if(diag <=left):
                if(diag<=top):
                    res = diag
                else:
                    res = top
            else:
                if(left<=top):
                    res = left
                else:
                    res = top
            cost_mat[i, j] = res+delta_mat[i, j]
    return cost_mat[s_len-1,t_len-1]



def fill_delta_mat_dtw(center, s, delta_mat, tmp_delta_mat):
    n_dims = center.shape[0]
    len_center = center.shape[1]
    len_s=  s.shape[1]
    slim = delta_mat[:len_center,:len_s]
    slim_tmp = tmp_delta_mat[:len_center,:len_s]

    #first dimension - not in the loop to avoid initialisation of delta_mat
    np.subtract.outer(center[0], s[0],out = slim)
    np.square(slim, out=slim)

    for d in range(1,center.shape[0]):
        np.subtract.outer(center[d], s[d],out = slim_tmp)
        np.square(slim_tmp, out=slim_tmp)
        np.add(slim,slim_tmp,out=slim)

    assert(np.abs(np.sum(np.square(center[:,0]-s[:,0]))-delta_mat[0,0])<=1e-6)


def DBA_update(center, series, cost_mat, path_mat, delta_mat, tmp_delta_mat):
    options_argmin = [(-1, -1), (0, -1), (-1, 0)]
    updated_center = np.zeros(center.shape)
    center_length = center.shape[1]
    n_elements = np.zeros(center_length, dtype=int)

    for s in series:
        s_len = s.shape[1]
        fill_delta_mat_dtw(center, s, delta_mat, tmp_delta_mat)
        cost_mat[0, 0] = delta_mat[0, 0]
        path_mat[0, 0] = -1

        for i in range(1, center_length):
            cost_mat[i, 0] = cost_mat[i-1, 0]+delta_mat[i, 0]
            path_mat[i, 0] = 2

        for j in range(1, s_len):
            cost_mat[0, j] = cost_mat[0, j-1]+delta_mat[0, j]
            path_mat[0, j] = 1

        for i in range(1, center_length):
            for j in range(1, s_len):
                diag,left,top =cost_mat[i-1, j-1], cost_mat[i, j-1], cost_mat[i-1, j]
                if(diag <=left):
                    if(diag<=top):
                        res = diag
                        path_mat[i,j] = 0
                    else:
                        res = top
                        path_mat[i,j] = 2
                else:
                    if(left<=top):
                        res = left
                        path_mat[i,j] = 1
                    else:
                        res = top
                        path_mat[i,j] = 2

                cost_mat[i, j] = res+delta_mat[i, j]

        i = center_length-1
        j = s_len-1

        while(path_mat[i, j] != -1):
            updated_center[:,i] += s[:,j]
            n_elements[i] += 1
            move = options_argmin[path_mat[i, j]]
            i += move[0]
            j += move[1]
        assert(i == 0 and j == 0)
        updated_center[:,i] += s[:,j]
        n_elements[i] += 1

    return np.divide(updated_center, n_elements)



