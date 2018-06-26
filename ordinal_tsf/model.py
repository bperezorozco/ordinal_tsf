from abc import ABCMeta, abstractmethod, abstractproperty
from keras import Model, Sequential, Input
from keras.layers import Dense, LSTM, Average, Bidirectional, Dropout, Concatenate
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
import pickle
import numpy as np
import GPy as gpy
import random
import os


class ModelStrategy(object):
    """Provides a common interface for the forecasting strategy to be used at runtime."""
    __metaclass__ = ABCMeta
    filename = 'tmp_'

    @abstractmethod
    def fit(self, dataset, **kwargs): pass

    @abstractmethod
    def predict(self, inputs, horizon=100, **kwargs): pass

    @staticmethod
    @abstractmethod
    def load(fname, **kwargs): pass

    @abstractmethod
    def save(self, folder): pass

    @staticmethod
    @abstractmethod
    def get_filename(params): pass

    @abstractproperty
    def seed_length(self): pass


class MordredStrategy(ModelStrategy):
    """Implements the ordinal sequence-to-sequence time series forecasting strategy."""
    required_spec_keys = ['ordinal_bins', 'units', 'dropout_rate', 'lam', 'horizon', 'lookback']
    id = 'mordred'

    def __init__(self, ordinal_bins=85, units=64, dropout_rate=0.25, lam=1e-9,
                 lookback=100, horizon=100, n_channels=1, custom_objs=[]):
        # type: (int, int, float, float, int, int, int, list) -> None
        self.n_bins = ordinal_bins
        self.n_hidden = units
        self.dropout_rate = dropout_rate
        self.lam = lam
        self.lookback = lookback
        self.horizon = horizon
        self.n_channels = n_channels
        self.filename = 'mordred_{}_bins_{}_hidden_{}_dropout_{}_l2_lookback_{}_horizon_{}_channels_{}'.format(
            self.n_bins, self.n_hidden, self.dropout_rate, self.lam, self.lookback, self.horizon, self.n_channels)

        loss = 'categorical_crossentropy'
        custom_objs = custom_objs

        lstm_spec = {'units': self.n_hidden,
                     'return_state': True,
                     'kernel_regularizer': l2(self.lam),
                     'recurrent_regularizer': l2(self.lam),
                     'dropout': self.dropout_rate,
                     'recurrent_dropout': self.dropout_rate}

        dense_spec = {'units': self.n_bins,
                      'activation': 'softmax',
                      'kernel_regularizer': l2(self.lam)}

        infr_init_h = Input(shape=(self.n_hidden,))
        infr_init_C = Input(shape=(self.n_hidden,))

        if self.n_channels > 1:
            all_encoder_inputs = [Input(shape=(None, self.n_bins), name='encoder_channel_{}'.format(i + 1))
                                  for i in range(self.n_channels)]
            all_decoder_inputs = [Input(shape=(None, self.n_bins), name='decoder_channel_{}'.format(i + 1))
                                  for i in range(self.n_channels)]

            encoder_input = Concatenate(axis=-1)(all_encoder_inputs)
            decoder_input = Concatenate(axis=-1)(all_decoder_inputs)
            train_inputs = all_encoder_inputs + all_decoder_inputs
            encoder_predict_inputs = all_encoder_inputs + [K.learning_phase()]
            decoder_predict_inputs = all_decoder_inputs + [infr_init_h, infr_init_C, K.learning_phase()]
        else:
            encoder_input = Input(shape=(None, self.n_bins))
            decoder_input = Input(shape=(None, self.n_bins))
            train_inputs = [encoder_input, decoder_input]
            encoder_predict_inputs = [encoder_input, K.learning_phase()]
            decoder_predict_inputs = [decoder_input, infr_init_h, infr_init_C, K.learning_phase()]

        encoder_fwd = LSTM(**lstm_spec)
        lstm_spec['go_backwards'] = True
        encoder_bkwd = LSTM(**lstm_spec)

        _, h_fwd, C_fwd = encoder_fwd(encoder_input)
        _, h_bkwd, C_bkwd = encoder_bkwd(encoder_input)
        decoder_initial_states = [Average()([h_fwd, h_bkwd]), Average()([C_fwd, C_bkwd])]

        lstm_spec['return_sequences'] = True
        lstm_spec['go_backwards'] = False
        decoder_lstm = LSTM(**lstm_spec)
        decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=decoder_initial_states)
        infr_decoder_output, infr_h, infr_C = decoder_lstm(decoder_input, initial_state=[infr_init_h, infr_init_C])

        if self.dropout_rate > 0.:
            decoder_output = Dropout(self.dropout_rate)(decoder_output)
            infr_decoder_output = Dropout(self.dropout_rate)(infr_decoder_output)

        if self.n_channels > 1:
            train_outputs = []
            decoder_predict_outputs = []

            for i in range(self.n_channels):
                decoder_dense = Dense(**dense_spec)
                train_outputs += [decoder_dense(decoder_output)]
                decoder_predict_outputs += [decoder_dense(infr_decoder_output)]

            decoder_predict_outputs += [infr_h, infr_C]
        else:
            decoder_dense = Dense(**dense_spec)
            decoded_sequence = decoder_dense(decoder_output)
            train_outputs = [decoded_sequence]
            infr_decoder_output, infr_h, infr_C = decoder_lstm(decoder_input, initial_state=[infr_init_h, infr_init_C])
            inferred_sequence = decoder_dense(infr_decoder_output)
            decoder_predict_outputs = [inferred_sequence, infr_h, infr_C]

        self.__sequence2sequence = Model(train_inputs, train_outputs)
        self.__sequence2sequence.compile(optimizer='nadam', loss=loss, metrics=[loss] + custom_objs)
        self.__encoder = Model(encoder_predict_inputs[:-1], decoder_initial_states)
        self.__decoder = Model(decoder_predict_inputs[:-1], decoder_predict_outputs)
        self.predict_stochastic = K.function(train_inputs + [K.learning_phase()], train_outputs)
        self.predict_stochastic_encoder = K.function(encoder_predict_inputs, decoder_initial_states)
        self.predict_stochastic_decoder = K.function(decoder_predict_inputs, decoder_predict_outputs)

    def fit(self, train_frames, **kwargs):
        # type: (np.ndarray) -> None
        cp_fname = 'cp_{}'.format(''.join([random.choice('0123456789ABCDEF') for _ in range(16)]))

        if train_frames.ndim > 3:
            inputs = [train_frames[:, :self.lookback, :, i] for i in range(train_frames.shape[-1])] + \
                     [train_frames[:, self.lookback:self.lookback + self.horizon, :, i]
                      for i in range(train_frames.shape[-1])]
            outputs = [train_frames[:, self.lookback + 1:self.lookback + self.horizon + 1, :, i]
                       for i in range(train_frames.shape[-1])]
        else:
            inputs = [train_frames[:, :self.lookback], train_frames[:, self.lookback:self.lookback + self.horizon]]
            outputs = [train_frames[:, self.lookback + 1:self.lookback + self.horizon + 1]]

        callbacks = [EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, verbose=1, mode='min'),
                     ModelCheckpoint(cp_fname, monitor='val_loss', mode='min',
                                     save_best_only=True,
                                     save_weights_only=True)]

        self.__sequence2sequence.fit(inputs, outputs, verbose=2, callbacks=callbacks, **kwargs)
        self.__sequence2sequence.load_weights(cp_fname)
        os.remove(cp_fname)

    def predict(self, inputs, predictive_horizon=100, mc_samples=100):
        samples = []
        if inputs.ndim > 3:
            encoder_inputs = [inputs[:, :self.lookback, :, i] for i in range(inputs.shape[3])]
            first_decoder_seed = [inputs[:, self.lookback:self.lookback + 1, :, i] for i in range(inputs.shape[3])]
        else:
            encoder_inputs = [inputs[:, :self.lookback]]
            first_decoder_seed = [inputs[:, self.lookback:self.lookback + 1]]

        for i_s in range(mc_samples):
            h, c = self.predict_stochastic_encoder(encoder_inputs + [True])
            decoder_stochastic_output = self.predict_stochastic_decoder(first_decoder_seed + [h, c, True])
            seq = [decoder_stochastic_output[:-2]]

            for t in range(predictive_horizon-1):
                decoder_stochastic_output = self.predict_stochastic_decoder(decoder_stochastic_output + [True])
                seq += [decoder_stochastic_output[:-2]]

            samples += [np.stack(seq, axis=-1).T.squeeze()]

        posterior_mean = np.stack(samples).mean(axis=0).squeeze()
        drawn_samples = []

        if self.n_channels > 1:
            for i_ch in range(self.n_channels):
                ch_posterior = posterior_mean.take(i_ch, axis=-1)
                ch_samples = [np.random.choice(self.n_bins, mc_samples, p=ch_posterior[t])
                                 for t in range(predictive_horizon)]
                drawn_samples += [np.stack(ch_samples, axis=-1)]
        else:
            drawn_samples += [np.random.choice(self.n_bins, mc_samples, p=posterior_mean[t])
                              for t in range(predictive_horizon)]

        drawn_samples = np.stack(drawn_samples, axis=-1)

        return {'ordinal_pdf': posterior_mean, 'draws': drawn_samples}

    def save(self, folder, fname=None):
        save_obj = {'ordinal_bins': self.n_bins,
                    'units': self.n_hidden,
                    'dropout_rate': self.dropout_rate,
                    'lam': self.lam,
                    'lookback': self.lookback,
                    'horizon': self.horizon,
                    'n_channels':self.n_channels}

        if fname is None:
            fname = MordredStrategy.get_filename(save_obj)

        fname = folder + fname
        weights_fname = fname + '_weights.h5'

        save_obj['weights_fname'] = weights_fname
        self.__sequence2sequence.save_weights(weights_fname, overwrite=True)

        with open(fname, 'wb') as f:
            pickle.dump(save_obj, f)

    def set_weights(self, weights_fname):
        self.__sequence2sequence.load_weights(weights_fname)

    @staticmethod
    def get_filename(model_spec):
        assert all([k in model_spec for k in MordredStrategy.required_spec_keys])
        return 'mordred_{}_bins_{}_hidden_{}_dropout_{}_l2_lookback_{}_horizon_{}_channels_{}'.format(model_spec['ordinal_bins'],
                                                                                                      model_spec['units'],
                                                                                                      model_spec['dropout_rate'],
                                                                                                      model_spec['lam'],
                                                                                                      model_spec['lookback'],
                                                                                                      model_spec['horizon'],
                                                                                                      model_spec['n_channels'])

    @staticmethod
    def load(fname, custom_objs = None):
        with open(fname, 'r') as f:
            spec = pickle.load(f)

        if custom_objs is not None:
            spec['custom_objs'] = custom_objs

        if 'lambda' in spec:
            l = spec.pop('lambda', 0.)
            spec['lam'] = l

        weights_fname = spec.pop('weights_fname', None)
        #print weights_fname
        assert weights_fname is not None, "Provide a valid weights filename to load model."

        model = MordredStrategy(**spec)
        model.set_weights(weights_fname)

        return model

    @property
    def seed_length(self):
        return self.lookback + 1


class GPStrategy(ModelStrategy):
    """Implements the autoregressive Gaussian Process time series forecasting strategy."""
    id = 'argp'

    def __init__(self, ker, lookback=100, horizon=1, fname='tmp', n_channels=1):
        self.ker = ker + gpy.kern.White(lookback)
        self.lookback = lookback
        self.horizon = horizon
        self.fname = 'gp_{}'.format(fname)  # TODO: DEFINE KERNEL STR
        self.model = None

    @staticmethod
    def load(fname):
        with open(fname, 'r') as f:
            obj = pickle.load(f)
        return obj

    def save(self, folder, fname=None):
        if fname is None:
            fname = self.fname

        with open(folder + fname, 'wb') as f:
            pickle.dump(self, f)

    def fit(self, train_frames, restarts=1):
        MAX_LENGTH = 10000
        self.model = gpy.models.GPRegression(train_frames[:MAX_LENGTH, :self.lookback, 0],  # TODO: attractor compatibility
                                             train_frames[:MAX_LENGTH, self.lookback:self.lookback+1, 0],
                                             self.ker)

        if restarts > 1:
            self.model.optimize_restarts(restarts)
        else:
            self.model.optimize()

    def predict(self, inputs, predictive_horizon=100, mc_samples=100):
        pred_inputs = inputs[:, :, 0]
        assert pred_inputs.ndim == 2  # TODO: reshape for attractor compatibility
        assert self.model is not None

        pred_mean, pred_var = self.model.predict(pred_inputs)
        pred_sigma = np.sqrt(pred_var)

        samples = np.random.normal(loc=pred_mean, scale=pred_sigma, size=(mc_samples, 1))
        draws = np.hstack((np.repeat(pred_inputs, axis=0, repeats=mc_samples), samples))

        for i in range(predictive_horizon - 1):
            pred_mu, pred_var = self.model.predict(draws[:, -self.seed_length:])
            pred_sigma = np.sqrt(pred_var)#.clip(0.) # TODO: sigma greater than 0
            samples = np.random.normal(loc=pred_mu, scale=pred_sigma)
            draws = np.hstack((draws, samples))

        return {'draws': draws[:, self.seed_length:]}  # TODO: attractor compatibility

    @staticmethod
    def get_filename(params):
        # type: (dict) -> str
        return 'gp_{}'.format(params.get('fname', 'tmp'))

    @property
    def seed_length(self):
        return self.lookback