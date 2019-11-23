from abc import ABCMeta, abstractmethod, abstractproperty
from keras import Model, Sequential, Input
from keras.layers import Dense, LSTM, Average, Bidirectional, Dropout, Concatenate
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from functools import partial
from pathlib import Path
import statsmodels.api as sm
import keras.backend as K
import pickle
import numpy as np
import GPy as gpy
import random
import gpflow as gpf
import gpflow.multioutput.features as mf
import uuid
import os
from ordinal_tsf.util import to_channels, to_contiguous

MAX_FNAME_LENGTH = 200
LONG_FNAMES_FNAME = 'long_fnames.txt'


class ModelStrategy(object):
    """Provides a common interface for the forecasting strategy to be used at runtime."""
    __metaclass__ = ABCMeta
    filename = 'tmp_'

    @abstractmethod
    def fit(self, train_frames, **kwargs): pass

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
        self.filename = '{}_{}_bins_{}_hidden_{}_dropout_{}_l2_lookback_{}_horizon_{}_channels_{}'.format(
            self.id, self.n_bins, self.n_hidden, self.dropout_rate, self.lam, self.lookback, self.horizon, self.n_channels)

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
            all_encoder_inputs = [Input(shape=(None, self.n_bins[i]), name='encoder_channel_{}'.format(i + 1))
                                  for i in range(self.n_channels)]
            all_decoder_inputs = [Input(shape=(None, self.n_bins[i]), name='decoder_channel_{}'.format(i + 1))
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
                dense_spec['units'] = self.n_bins[i]
                decoder_dense = Dense(**dense_spec)
                train_outputs += [decoder_dense(decoder_output)]
                decoder_predict_outputs += [decoder_dense(infr_decoder_output)]

            decoder_predict_outputs += [infr_h, infr_C]
        else:
            decoder_dense = Dense(**dense_spec)
            decoded_sequence = decoder_dense(decoder_output)
            train_outputs = [decoded_sequence]
            inferred_sequence = decoder_dense(infr_decoder_output)
            decoder_predict_outputs = [inferred_sequence, infr_h, infr_C]

        self.__sequence2sequence = Model(train_inputs, train_outputs)
        self.__sequence2sequence.compile(optimizer='nadam', loss=loss, metrics=[loss] + custom_objs)
        self.__encoder = Model(encoder_predict_inputs[:-1], decoder_initial_states) # no learning phase
        self.__decoder = Model(decoder_predict_inputs[:-1], decoder_predict_outputs)
        self.predict_stochastic = K.function(train_inputs + [K.learning_phase()], train_outputs)
        self.predict_stochastic_encoder = K.function(encoder_predict_inputs, decoder_initial_states)
        self.predict_stochastic_decoder = K.function(decoder_predict_inputs, decoder_predict_outputs)

    def fit(self, train_frames, **kwargs):
        # IMPORTANT: asssumes train_frames is a nparray which technically
        # does not allow for channels with different number of bins
        batch_size = kwargs.get('batch_size', 256)
        val_p = kwargs.get('validation_split', 0.15)
        epochs = kwargs.get('epochs', 50)

        def get_inputs(x):
            if x.ndim > 3:
                return [x[:, :self.lookback, :, i] for i in range(x.shape[-1])] + \
                       [x[:, self.lookback:self.lookback + self.horizon, :, i]
                        for i in range(x.shape[-1])]
            else:
                return [x[:, :self.lookback], x[:, self.lookback:self.lookback + self.horizon]]

        def get_outputs(x):
            if x.ndim > 3:
                return [x[:, self.lookback + 1:self.lookback + self.horizon + 1, :, i]
                        for i in range(x.shape[-1])]
            else:
                return [x[:, self.lookback + 1:self.lookback + self.horizon + 1]]

        train_gen, val_gen, tr_steps, val_steps = train_frames(get_inputs=get_inputs, get_outputs=get_outputs,
                                                               batch_size=batch_size, val_p=val_p)

        cp_fname = 'cp_{}'.format(''.join([random.choice('0123456789ABCDEF') for _ in range(16)]))

        callbacks = [EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, verbose=1, mode='min'),
                     ModelCheckpoint(cp_fname, monitor='val_loss', mode='min',
                                     save_best_only=True,
                                     save_weights_only=True)]

        self.__sequence2sequence.fit_generator(train_gen,
                                               steps_per_epoch=tr_steps,
                                               verbose=2,
                                               validation_data=val_gen,
                                               validation_steps=val_steps,
                                               callbacks=callbacks,
                                               epochs=epochs)

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
                ch_samples = [np.random.choice(self.n_bins[i_ch], mc_samples, p=ch_posterior[t])
                                 for t in range(predictive_horizon)]
                drawn_samples += [np.stack(ch_samples, axis=-1)]
        else:
            drawn_samples += [np.random.choice(self.n_bins, mc_samples, p=posterior_mean[t])
                              for t in range(predictive_horizon)]

        drawn_samples = np.stack(drawn_samples, axis=-1)

        return {'ordinal_pdf': posterior_mean, 'draws': drawn_samples}

    def save(self, folder, fname=None):
        if isinstance(self.n_bins, (list,)):
            ord_bins = '_'.join(['chbins{}_{}'.format(i_ch+1, b) for i_ch, b in enumerate(self.n_bins)])
        else:
            ord_bins = self.n_bins

        save_obj = {'ordinal_bins': ord_bins,
                    'units': self.n_hidden,
                    'dropout_rate': self.dropout_rate,
                    'lam': self.lam,
                    'lookback': self.lookback,
                    'horizon': self.horizon,
                    'n_channels':self.n_channels}

        if fname is None:
            fname = MordredStrategy.get_filename(save_obj, folder)

        fname = folder + fname
        weights_fname = fname + '_weights.h5'

        save_obj['weights_fname'] = weights_fname
        self.__sequence2sequence.save_weights(weights_fname, overwrite=True)

        with open(fname, 'wb') as f:
            pickle.dump(save_obj, f)

    def set_weights(self, weights_fname):
        self.__sequence2sequence.load_weights(weights_fname)

    @staticmethod
    def get_filename(model_spec, folder='.'):
        assert all([k in model_spec for k in MordredStrategy.required_spec_keys])

        if isinstance(model_spec['ordinal_bins'], (list,)):
            ord_bins = '_'.join(['chbins{}_{}'.format(i_ch+1, b) for i_ch, b in enumerate(model_spec['ordinal_bins'])])
        else:
            ord_bins = model_spec['ordinal_bins']

        fname = 'mordred_{}_bins_{}_hidden_{}_dropout_{}_l2_lookback_{}_horizon_{}_channels_{}'.format(ord_bins,
                                                                                                      model_spec['units'],
                                                                                                      model_spec['dropout_rate'],
                                                                                                      model_spec['lam'],
                                                                                                      model_spec['lookback'],
                                                                                                      model_spec['horizon'],
                                                                                                      model_spec['n_channels'])


        return fname[:MAX_FNAME_LENGTH]

    @staticmethod
    def load(fname, custom_objs = None):
        with open(fname, 'rb') as f:
            spec = pickle.load(f)

        if custom_objs is not None:
            spec['custom_objs'] = custom_objs

        if 'lambda' in spec:
            l = spec.pop('lambda', 0.)
            spec['lam'] = l

        weights_fname = spec.pop('weights_fname', None)

        if type(spec['ordinal_bins']) is not int:
            spec['ordinal_bins'] = [int(i) for i in spec['ordinal_bins'].split('_')[1:][::2]]
        #print(weights_fname)
        assert weights_fname is not None, "Provide a valid weights filename to load model."

        model = MordredStrategy(**spec)
        model.set_weights(weights_fname)

        return model

    @property
    def seed_length(self):
        return self.lookback + 1


class MordredAutoencoderStrategy(MordredStrategy):
    id = 'mordred_autoencoder'

    def fit(self, train_frames, **kwargs):
        # IMPORTANT: asssumes train_frames is a nparray which technically
        # does not allow for channels with different number of bins
        batch_size = kwargs.get('batch_size', 256)
        val_p = kwargs.get('validation_split', 0.15)
        epochs = kwargs.get('epochs', 50)

        def get_inputs(x):
            if x.ndim > 3:
                return [x[:, :self.lookback, :, i] for i in range(x.shape[-1])] + \
                       [x[:, : self.lookback, :, i]
                        for i in range(x.shape[-1])]
            else:
                return [x[:, :self.lookback], x[:, :self.lookback ]]

        def get_outputs(x):
            if x.ndim > 3:
                return [x[:, :self.lookback, :, i] for i in range(x.shape[-1])]
            else:
                return [x[:, :self.lookback]]

        train_gen, val_gen, tr_steps, val_steps = train_frames(get_inputs=get_inputs, get_outputs=get_outputs,
                                                               batch_size=batch_size, val_p=val_p)

        cp_fname = 'cp_{}'.format(''.join([random.choice('0123456789ABCDEF') for _ in range(16)]))

        callbacks = [EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, verbose=1, mode='min'),
                     ModelCheckpoint(cp_fname, monitor='val_loss', mode='min',
                                     save_best_only=True,
                                     save_weights_only=True)]

        self.__sequence2sequence.fit_generator(train_gen,
                                               steps_per_epoch=tr_steps,
                                               verbose=2,
                                               validation_data=val_gen,
                                               validation_steps=val_steps,
                                               callbacks=callbacks,
                                               epochs=epochs)

        self.__sequence2sequence.load_weights(cp_fname)
        os.remove(cp_fname)


class MultilayerMordredStrategy(ModelStrategy):
    """Implements the ordinal sequence-to-sequence time series forecasting strategy."""
    required_spec_keys = ['ordinal_bins', 'units', 'n_layers', 'dropout_rate', 'lam', 'horizon', 'lookback']
    id = 'multilayer_mordred'

    def __init__(self, ordinal_bins=85, n_layers=2, units=64, dropout_rate=0.25, lam=1e-9,
                 lookback=100, horizon=100, n_channels=1, custom_objs=[]):
        # type: (int, int, float, float, int, int, int, list) -> None
        self.n_bins = ordinal_bins
        self.n_hidden = units
        self.dropout_rate = dropout_rate
        self.lam = lam
        self.lookback = lookback
        self.horizon = horizon
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.filename = '{}_{}_bins_{}_hidden_{}_dropout_{}_l2_lookback_{}_horizon_{}_channels_{}'.format(
            self.id, self.n_bins, self.n_hidden, self.dropout_rate, self.lam, self.lookback, self.horizon, self.n_channels)

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
            all_encoder_inputs = [Input(shape=(None, self.n_bins[i]), name='encoder_channel_{}'.format(i + 1))
                                  for i in range(self.n_channels)]
            all_decoder_inputs = [Input(shape=(None, self.n_bins[i]), name='decoder_channel_{}'.format(i + 1))
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

        lstm_spec['return_sequences'] = False
        encoder_fwd = LSTM(**lstm_spec)

        lstm_spec['go_backwards'] = True
        encoder_bkwd = LSTM(**lstm_spec)

        lstm_spec['return_sequences'] = True
        prev_encoder_bkwd = LSTM(**lstm_spec)

        lstm_spec['go_backwards'] = False
        prev_encoder_fwd = LSTM(**lstm_spec)

        _, h_fwd, C_fwd = encoder_fwd(prev_encoder_fwd(encoder_input))
        _, h_bkwd, C_bkwd = encoder_bkwd(prev_encoder_bkwd(encoder_input))

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
                dense_spec['units'] = self.n_hidden
                prev_decoder = Dense(**dense_spec)
                dense_spec['units'] = self.n_bins[i]
                decoder_dense = Dense(**dense_spec)
                train_outputs += [decoder_dense(prev_decoder(decoder_output))]
                decoder_predict_outputs += [decoder_dense(prev_decoder(infr_decoder_output))]

            decoder_predict_outputs += [infr_h, infr_C]
        else:
            dense_spec['units'] = self.n_hidden
            prev_decoder = Dense(**dense_spec)
            dense_spec['units'] = self.n_bins
            decoder_dense = Dense(**dense_spec)
            decoded_sequence = decoder_dense(prev_decoder(decoder_output))
            train_outputs = [decoded_sequence]
            inferred_sequence = decoder_dense(prev_decoder(infr_decoder_output))
            decoder_predict_outputs = [inferred_sequence, infr_h, infr_C]

        self.__sequence2sequence = Model(train_inputs, train_outputs)
        self.__sequence2sequence.compile(optimizer='nadam', loss=loss, metrics=[loss] + custom_objs)
        self.__encoder = Model(encoder_predict_inputs[:-1], decoder_initial_states) # no learning phase
        self.__decoder = Model(decoder_predict_inputs[:-1], decoder_predict_outputs)
        self.predict_stochastic = K.function(train_inputs + [K.learning_phase()], train_outputs)
        self.predict_stochastic_encoder = K.function(encoder_predict_inputs, decoder_initial_states)
        self.predict_stochastic_decoder = K.function(decoder_predict_inputs, decoder_predict_outputs)

    def fit(self, train_frames, **kwargs):
        # IMPORTANT: asssumes train_frames is a nparray which technically
        # does not allow for channels with different number of bins
        batch_size = kwargs.get('batch_size', 256)
        val_p = kwargs.get('validation_split', 0.15)
        epochs = kwargs.get('epochs', 50)

        def get_inputs(x):
            if x.ndim > 3:
                return [x[:, :self.lookback, :, i] for i in range(x.shape[-1])] + \
                       [x[:, self.lookback:self.lookback + self.horizon, :, i]
                        for i in range(x.shape[-1])]
            else:
                return [x[:, :self.lookback], x[:, self.lookback:self.lookback + self.horizon]]

        def get_outputs(x):
            if x.ndim > 3:
                return [x[:, self.lookback + 1:self.lookback + self.horizon + 1, :, i]
                        for i in range(x.shape[-1])]
            else:
                return [x[:, self.lookback + 1:self.lookback + self.horizon + 1]]

        train_gen, val_gen, tr_steps, val_steps = train_frames(get_inputs=get_inputs, get_outputs=get_outputs,
                                                               batch_size=batch_size, val_p=val_p)

        cp_fname = 'cp_{}'.format(''.join([random.choice('0123456789ABCDEF') for _ in range(16)]))

        callbacks = [EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, verbose=1, mode='min'),
                     ModelCheckpoint(cp_fname, monitor='val_loss', mode='min',
                                     save_best_only=True,
                                     save_weights_only=True)]

        self.__sequence2sequence.fit_generator(train_gen,
                                               steps_per_epoch=tr_steps,
                                               verbose=2,
                                               validation_data=val_gen,
                                               validation_steps=val_steps,
                                               callbacks=callbacks,
                                               epochs=epochs)

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
                ch_samples = [np.random.choice(self.n_bins[i_ch], mc_samples, p=ch_posterior[t])
                                 for t in range(predictive_horizon)]
                drawn_samples += [np.stack(ch_samples, axis=-1)]
        else:
            drawn_samples += [np.random.choice(self.n_bins, mc_samples, p=posterior_mean[t])
                              for t in range(predictive_horizon)]

        drawn_samples = np.stack(drawn_samples, axis=-1)

        return {'ordinal_pdf': posterior_mean, 'draws': drawn_samples}

    def save(self, folder, fname=None):
        if isinstance(self.n_bins, (list,)):
            ord_bins = '_'.join(['chbins{}_{}'.format(i_ch+1, b) for i_ch, b in enumerate(self.n_bins)])
        else:
            ord_bins = self.n_bins

        save_obj = {'ordinal_bins': ord_bins,
                    'units': self.n_hidden,
                    'dropout_rate': self.dropout_rate,
                    'lam': self.lam,
                    'lookback': self.lookback,
                    'horizon': self.horizon,
                    'n_channels':self.n_channels,
                    'n_layers': self.n_layers}

        if fname is None:
            fname = MultilayerMordredStrategy.get_filename(save_obj, folder)

        fname = folder + fname
        weights_fname = fname + '_weights.h5'

        save_obj['weights_fname'] = weights_fname
        self.__sequence2sequence.save_weights(weights_fname, overwrite=True)

        with open(fname, 'wb') as f:
            pickle.dump(save_obj, f)

    def set_weights(self, weights_fname):
        self.__sequence2sequence.load_weights(weights_fname)

    @staticmethod
    def get_filename(model_spec, folder='.'):
        assert all([k in model_spec for k in MultilayerMordredStrategy.required_spec_keys])

        if isinstance(model_spec['ordinal_bins'], (list,)):
            ord_bins = '_'.join(['chbins{}_{}'.format(i_ch+1, b) for i_ch, b in enumerate(model_spec['ordinal_bins'])])
        else:
            ord_bins = model_spec['ordinal_bins']

        fname = 'multilayer_mordred_{}_bins_{}_hidden_{}_layers_{}_dropout_{}_l2_lookback_{}_horizon_{}_channels_{}'.format(ord_bins,
                                                                                                      model_spec['units'],
                                                                                                      model_spec['n_layers'],
                                                                                                      model_spec['dropout_rate'],
                                                                                                      model_spec['lam'],
                                                                                                      model_spec['lookback'],
                                                                                                      model_spec['horizon'],
                                                                                                      model_spec['n_channels'])

        return fname[:MAX_FNAME_LENGTH]

    @staticmethod
    def load(fname, custom_objs = None):
        with open(fname, 'rb') as f:
            spec = pickle.load(f)

        if custom_objs is not None:
            spec['custom_objs'] = custom_objs

        if 'lambda' in spec:
            l = spec.pop('lambda', 0.)
            spec['lam'] = l

        weights_fname = spec.pop('weights_fname', None)

        if type(spec['ordinal_bins']) is not int:
            spec['ordinal_bins'] = [int(i) for i in spec['ordinal_bins'].split('_')[1:][::2]]
        #print(weights_fname)
        assert weights_fname is not None, "Provide a valid weights filename to load model."

        model = MultilayerMordredStrategy(**spec)
        model.set_weights(weights_fname)

        return model

    @property
    def seed_length(self):
        return self.lookback + 1


class AttentionMordredStrategy(ModelStrategy):
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
        self.filename = '{}_{}_bins_{}_hidden_{}_dropout_{}_l2_lookback_{}_horizon_{}_channels_{}'.format(
            self.id, self.n_bins, self.n_hidden, self.dropout_rate, self.lam, self.lookback, self.horizon, self.n_channels)

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
            all_encoder_inputs = [Input(shape=(None, self.n_bins[i]), name='encoder_channel_{}'.format(i + 1))
                                  for i in range(self.n_channels)]
            all_decoder_inputs = [Input(shape=(None, self.n_bins[i]), name='decoder_channel_{}'.format(i + 1))
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
                dense_spec['units'] = self.n_bins[i]
                decoder_dense = Dense(**dense_spec)
                train_outputs += [decoder_dense(decoder_output)]
                decoder_predict_outputs += [decoder_dense(infr_decoder_output)]

            decoder_predict_outputs += [infr_h, infr_C]
        else:
            decoder_dense = Dense(**dense_spec)
            decoded_sequence = decoder_dense(decoder_output)
            train_outputs = [decoded_sequence]
            inferred_sequence = decoder_dense(infr_decoder_output)
            decoder_predict_outputs = [inferred_sequence, infr_h, infr_C]

        self.__sequence2sequence = Model(train_inputs, train_outputs)
        self.__sequence2sequence.compile(optimizer='nadam', loss=loss, metrics=[loss] + custom_objs)
        self.__encoder = Model(encoder_predict_inputs[:-1], decoder_initial_states) # no learning phase
        self.__decoder = Model(decoder_predict_inputs[:-1], decoder_predict_outputs)
        self.predict_stochastic = K.function(train_inputs + [K.learning_phase()], train_outputs)
        self.predict_stochastic_encoder = K.function(encoder_predict_inputs, decoder_initial_states)
        self.predict_stochastic_decoder = K.function(decoder_predict_inputs, decoder_predict_outputs)

    def fit(self, train_frames, **kwargs):
        # IMPORTANT: asssumes train_frames is a nparray which technically
        # does not allow for channels with different number of bins
        batch_size = kwargs.get('batch_size', 256)
        val_p = kwargs.get('validation_split', 0.15)
        epochs = kwargs.get('epochs', 50)

        def get_inputs(x):
            if x.ndim > 3:
                return [x[:, :self.lookback, :, i] for i in range(x.shape[-1])] + \
                       [x[:, self.lookback:self.lookback + self.horizon, :, i]
                        for i in range(x.shape[-1])]
            else:
                return [x[:, :self.lookback], x[:, self.lookback:self.lookback + self.horizon]]

        def get_outputs(x):
            if x.ndim > 3:
                return [x[:, self.lookback + 1:self.lookback + self.horizon + 1, :, i]
                        for i in range(x.shape[-1])]
            else:
                return [x[:, self.lookback + 1:self.lookback + self.horizon + 1]]

        train_gen, val_gen, tr_steps, val_steps = train_frames(get_inputs=get_inputs, get_outputs=get_outputs,
                                                               batch_size=batch_size, val_p=val_p)

        cp_fname = 'cp_{}'.format(''.join([random.choice('0123456789ABCDEF') for _ in range(16)]))

        callbacks = [EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, verbose=1, mode='min'),
                     ModelCheckpoint(cp_fname, monitor='val_loss', mode='min',
                                     save_best_only=True,
                                     save_weights_only=True)]

        self.__sequence2sequence.fit_generator(train_gen,
                                               steps_per_epoch=tr_steps,
                                               verbose=2,
                                               validation_data=val_gen,
                                               validation_steps=val_steps,
                                               callbacks=callbacks,
                                               epochs=epochs)

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
                ch_samples = [np.random.choice(self.n_bins[i_ch], mc_samples, p=ch_posterior[t])
                                 for t in range(predictive_horizon)]
                drawn_samples += [np.stack(ch_samples, axis=-1)]
        else:
            drawn_samples += [np.random.choice(self.n_bins, mc_samples, p=posterior_mean[t])
                              for t in range(predictive_horizon)]

        drawn_samples = np.stack(drawn_samples, axis=-1)

        return {'ordinal_pdf': posterior_mean, 'draws': drawn_samples}

    def save(self, folder, fname=None):
        if isinstance(self.n_bins, (list,)):
            ord_bins = '_'.join(['chbins{}_{}'.format(i_ch+1, b) for i_ch, b in enumerate(self.n_bins)])
        else:
            ord_bins = self.n_bins

        save_obj = {'ordinal_bins': ord_bins,
                    'units': self.n_hidden,
                    'dropout_rate': self.dropout_rate,
                    'lam': self.lam,
                    'lookback': self.lookback,
                    'horizon': self.horizon,
                    'n_channels':self.n_channels}

        if fname is None:
            fname = MordredStrategy.get_filename(save_obj, folder)

        fname = folder + fname
        weights_fname = fname + '_weights.h5'

        save_obj['weights_fname'] = weights_fname
        self.__sequence2sequence.save_weights(weights_fname, overwrite=True)

        with open(fname, 'wb') as f:
            pickle.dump(save_obj, f)

    def set_weights(self, weights_fname):
        self.__sequence2sequence.load_weights(weights_fname)

    @staticmethod
    def get_filename(model_spec, folder='.'):
        assert all([k in model_spec for k in MordredStrategy.required_spec_keys])

        if isinstance(model_spec['ordinal_bins'], (list,)):
            ord_bins = '_'.join(['chbins{}_{}'.format(i_ch+1, b) for i_ch, b in enumerate(model_spec['ordinal_bins'])])
        else:
            ord_bins = model_spec['ordinal_bins']

        fname = 'mordred_{}_bins_{}_hidden_{}_dropout_{}_l2_lookback_{}_horizon_{}_channels_{}'.format(ord_bins,
                                                                                                      model_spec['units'],
                                                                                                      model_spec['dropout_rate'],
                                                                                                      model_spec['lam'],
                                                                                                      model_spec['lookback'],
                                                                                                      model_spec['horizon'],
                                                                                                      model_spec['n_channels'])


        return fname[:MAX_FNAME_LENGTH]

    @staticmethod
    def load(fname, custom_objs = None):
        with open(fname, 'rb') as f:
            spec = pickle.load(f)

        if custom_objs is not None:
            spec['custom_objs'] = custom_objs

        if 'lambda' in spec:
            l = spec.pop('lambda', 0.)
            spec['lam'] = l

        weights_fname = spec.pop('weights_fname', None)

        if type(spec['ordinal_bins']) is not int:
            spec['ordinal_bins'] = [int(i) for i in spec['ordinal_bins'].split('_')[1:][::2]]
        #print(weights_fname)
        assert weights_fname is not None, "Provide a valid weights filename to load model."

        model = MordredStrategy(**spec)
        model.set_weights(weights_fname)

        return model

    @property
    def seed_length(self):
        return self.lookback + 1


class MordredXStrategy(ModelStrategy):
    required_spec_keys = ['n_ar_channels', 'n_exog_channels', 'units', 'dropout_rate', 'lam', 'horizon', 'lookback']
    id = 'mordredX'

    def __init__(self, ar_ordinal_bins=85, exog_ordinal_bins=85, units=64, dropout_rate=0.25, lam=1e-9,
                 lookback=100, horizon=100, n_ar_channels=1, n_exog_channels=1, custom_objs=[]):
        # type: (int, int, float, float, int, int, int, list) -> None
        self.n_ar_bins = ar_ordinal_bins
        self.n_exog_bins = exog_ordinal_bins
        self.n_hidden = units
        self.dropout_rate = dropout_rate
        self.lam = lam
        self.lookback = lookback
        self.horizon = horizon
        self.n_ar_channels = n_ar_channels
        self.n_exog_channels = n_exog_channels

        #self.filename = 'mordredx_{}_bins_{}_hidden_{}_dropout_{}_l2_lookback_{}_horizon_{}_ARchannels_{}_EXchannels_{}'\
        #    .format(self.n_bins, self.n_hidden, self.dropout_rate, self.lam,
        #            self.lookback, self.horizon, self.n_ar_channels, self.n_exog_channels)

        loss = 'categorical_crossentropy'
        custom_objs = custom_objs

        lstm_spec = {'units': self.n_hidden,
                     'return_state': True,
                     'kernel_regularizer': l2(self.lam),
                     'recurrent_regularizer': l2(self.lam),
                     'dropout': self.dropout_rate,
                     'recurrent_dropout': self.dropout_rate}

        dense_spec = {'activation': 'softmax',
                      'kernel_regularizer': l2(self.lam)}

        infr_init_h = Input(shape=(self.n_hidden,))
        infr_init_C = Input(shape=(self.n_hidden,))

        all_encoder_inputs = [Input(shape=(None, self.n_ar_bins[i]), name='encoder_channel_{}'.format(i + 1))
                              for i in range(self.n_ar_channels)]
        all_exog_encoder_inputs = [Input(shape=(None, self.n_exog_bins[i]), name='exog_encoder_channel_{}'.format(i + 1))
                              for i in range(self.n_exog_channels)]
        all_decoder_inputs = [Input(shape=(None, self.n_ar_bins[i]), name='decoder_channel_{}'.format(i + 1))
                              for i in range(self.n_ar_channels)]
        all_exog_decoder_inputs = [Input(shape=(None, self.n_exog_bins[i]), name='exog_decoder_channel_{}'.format(i + 1))
                                   for i in range(self.n_exog_channels)]

        encoder_input = Concatenate(axis=-1)(all_encoder_inputs + all_exog_encoder_inputs)
        decoder_input = Concatenate(axis=-1)(all_decoder_inputs + all_exog_decoder_inputs)
        train_inputs = all_encoder_inputs + all_exog_encoder_inputs + all_decoder_inputs + all_exog_decoder_inputs
        encoder_predict_inputs = all_encoder_inputs + all_exog_encoder_inputs + [K.learning_phase()]
        decoder_predict_inputs = all_decoder_inputs + all_exog_decoder_inputs + [infr_init_h,
                                                                                 infr_init_C,
                                                                                 K.learning_phase()]

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

        train_outputs = []
        decoder_predict_outputs = []

        for i in range(self.n_ar_channels):
            dense_spec['units'] = self.n_ar_bins[i]
            decoder_dense = Dense(**dense_spec)
            train_outputs += [decoder_dense(decoder_output)]
            decoder_predict_outputs += [decoder_dense(infr_decoder_output)]

        decoder_predict_outputs += [infr_h, infr_C]

        self.__sequence2sequence = Model(train_inputs, train_outputs)
        self.__sequence2sequence.compile(optimizer='nadam', loss=loss, metrics=[loss] + custom_objs)
        self.__encoder = Model(encoder_predict_inputs[:-1], decoder_initial_states)
        self.__decoder = Model(decoder_predict_inputs[:-1], decoder_predict_outputs)
        self.predict_stochastic = K.function(train_inputs + [K.learning_phase()], train_outputs)
        self.predict_stochastic_encoder = K.function(encoder_predict_inputs, decoder_initial_states)
        self.predict_stochastic_decoder = K.function(decoder_predict_inputs, decoder_predict_outputs)

    def fit(self, train_frames, **kwargs):
        # IMPORTANT: asssumes train_frames is a nparray which technically
        # does not allow for channels with different number of bins
        # output channels come before exogenous channels
        batch_size = kwargs.get('batch_size', 256)
        val_p = kwargs.get('validation_split', 0.15)
        epochs = kwargs.get('epochs', 50)

        def get_inputs(x_list):
            return [x[:, :self.lookback] for x in x_list] + \
                   [x[:, self.lookback:self.lookback + self.horizon]
                    for x in x_list]

        def get_outputs(x_list, n_ar=1):
            return [x[:, self.lookback + 1:self.lookback + self.horizon + 1]
                    for x in x_list[:n_ar]]

        train_gen, val_gen, tr_steps, val_steps = train_frames(get_inputs=get_inputs,
                                                               get_outputs=partial(get_outputs, n_ar=self.n_ar_channels),
                                                               batch_size=batch_size, val_p=val_p)

        cp_fname = 'cp_{}'.format(''.join([random.choice('0123456789ABCDEF') for _ in range(16)]))

        callbacks = [EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, verbose=1, mode='min'),
                     ModelCheckpoint(cp_fname, monitor='val_loss', mode='min',
                                     save_best_only=True,
                                     save_weights_only=True)]

        self.__sequence2sequence.fit_generator(train_gen,
                                               steps_per_epoch=tr_steps,
                                               verbose=2,
                                               validation_data=val_gen,
                                               validation_steps=val_steps,
                                               callbacks=callbacks,
                                               epochs=epochs)

        self.__sequence2sequence.load_weights(cp_fname)
        os.remove(cp_fname)

    def predict(self, ar_input_list, exog_input_list=[], predictive_horizon=100, mc_samples=100):
        #exog_input_list[i] must have at least lookback + predictive_horizon samples
        exog_min_length = self.lookback + predictive_horizon

        for i_exog, exog_input in enumerate(exog_input_list):
            assert exog_input.shape[1] >= exog_min_length, '{} exog input has {} < {} samples'.format(i_exog,
                                                                                                      exog_input.shape[1],
                                                                                                      exog_min_length)
        samples = [[] for _ in range(self.n_ar_channels)]
        encoder_inputs = [inputs[:, :self.lookback, :] for inputs in ar_input_list + exog_input_list]
        first_decoder_seed = [inputs[:, self.lookback:self.lookback+1, :] for inputs in ar_input_list + exog_input_list]

        for i_s in range(mc_samples):
            h, c = self.predict_stochastic_encoder(encoder_inputs + [True])
            decoder_stochastic_output = self.predict_stochastic_decoder(first_decoder_seed + [h, c, True])
            seq = [decoder_stochastic_output[:-2]] # length is number of AR channels

            for t in range(1, predictive_horizon):
                current_exog_input = [inputs[:, self.lookback+t:self.lookback+t+1, :] for inputs in exog_input_list]
                decoder_stochastic_output = self.predict_stochastic_decoder(decoder_stochastic_output[:-2]
                                                                            + current_exog_input
                                                                            + decoder_stochastic_output[-2:]
                                                                            + [True])
                seq += [decoder_stochastic_output[:-2]]

            for i_ch in range(self.n_ar_channels):
                samples[i_ch] += [np.stack([s[i_ch] for s in seq], axis=-1).T.squeeze()]

        posterior_mean = [np.stack(i_samples).mean(axis=0).squeeze() for i_samples in samples]
        drawn_samples = []

        for i_ch in range(self.n_ar_channels):
            ch_posterior = posterior_mean[i_ch]
            ch_samples = [np.random.choice(self.n_ar_bins[i_ch], mc_samples, p=ch_posterior[t])
                          for t in range(predictive_horizon)]
            drawn_samples += [np.stack(ch_samples, axis=-1)]

        return {'ordinal_pdf': posterior_mean, 'draws': drawn_samples}

    def set_weights(self, weights_fname):
        self.__sequence2sequence.load_weights(weights_fname)

    @staticmethod
    def load(fname, custom_objs = None):
        with open(fname, 'rb') as f:
            spec = pickle.load(f)

        if custom_objs is not None:
            spec['custom_objs'] = custom_objs

        if 'lambda' in spec:
            l = spec.pop('lambda', 0.)
            spec['lam'] = l

        weights_fname = spec.pop('weights_fname', None)

        assert weights_fname is not None, "Provide a valid weights filename to load model."

        model = MordredXStrategy(**spec)
        model.set_weights(weights_fname)

        return model

    def save(self, folder, fname=None):
        save_obj = {'units': self.n_hidden,
                    'dropout_rate': self.dropout_rate,
                    'lam': self.lam,
                    'lookback': self.lookback,
                    'horizon': self.horizon,
                    'n_ar_channels': self.n_ar_channels,
                    'n_exog_channels': self.n_exog_channels,
                    'ar_ordinal_bins':self.n_ar_bins,
                    'exog_ordinal_bins':self.n_exog_bins}

        if fname is None:
            fname = MordredXStrategy.get_filename(save_obj, folder)

        fname = folder + fname
        weights_fname = fname + '_weights.h5'

        save_obj['weights_fname'] = weights_fname
        self.__sequence2sequence.save_weights(weights_fname, overwrite=True)

        with open(fname, 'wb') as f:
            pickle.dump(save_obj, f)

    def get_spec(self):
        return {'units': self.n_hidden,
                'dropout_rate': self.dropout_rate,
                'lam': self.lam,
                'lookback': self.lookback,
                'horizon': self.horizon,
                'n_ar_channels': self.n_ar_channels,
                'n_exog_channels': self.n_exog_channels,
                'ar_ordinal_bins':self.n_ar_bins,
                'exog_ordinal_bins':self.n_exog_bins}

    @staticmethod
    def get_filename(model_spec, folder='.'):
        assert all([k in model_spec for k in MordredXStrategy.required_spec_keys])

        fname = 'mordredx_{}_hidden_{}_dropout_{}_l2_lookback_{}_horizon_{}_channels_{}_exog_{}'.format(model_spec[
                                                                                                           'units'],
                                                                                                       model_spec[
                                                                                                           'dropout_rate'],
                                                                                                       model_spec[
                                                                                                           'lam'],
                                                                                                       model_spec[
                                                                                                           'lookback'],
                                                                                                       model_spec[
                                                                                                           'horizon'],
                                                                                                       model_spec[
                                                                                                           'n_ar_channels'],
                                                                                                       model_spec[
                                                                                                           'n_exog_channels']
                                                                                                       )

        return fname[:MAX_FNAME_LENGTH]

    @property
    def seed_length(self):
        return self.lookback + 1


class SARIMAXStrategy(ModelStrategy):
    filename = ''
    id = 'sarimax'

    def __init__(self, order, seasonal_order=(0,0,0,0)):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, train_frames, **kwargs):
        self.model = sm.tsa.statespace.SARIMAX(train_frames,
                                               order=self.order,
                                               seasonal_order=self.seasonal_order,
                                               enforce_stationarity=False)
        self.fit_res = self.model.fit(disp=False)

    def predict(self, inputs, predictive_horizon=100, **kwargs):
        pred = self.fit_res.get_forecast(steps=predictive_horizon)
        return {'posterior_mean':pred.predicted_mean, 'posterior_std':np.sqrt(pred.var_pred_mean)}

    @staticmethod
    def load(fname, **kwargs):
        this = None
        with open(fname, 'r') as f:
            this = pickle.load(f)

        return this

    def save(self, folder):
        params = {'order':self.order, 'seasonal_order':self.seasonal_order}

        with open(folder + SARIMAXStrategy.get_filename(params), 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def get_filename(params):
        # type: (dict) -> str
        return 'sarimax_{}_{}'.format(params['order'][0], params['seasonal_order'][0])

    @property
    def seed_length(self):
        return 121


class ContinuousSeq2Seq(ModelStrategy):
    """Implements the ordinal sequence-to-sequence time series forecasting strategy."""
    required_spec_keys = ['units', 'dropout_rate', 'lam', 'horizon', 'lookback']
    id = 'seq2seq'

    def __init__(self, units=64, dropout_rate=0.25, lam=1e-9,
                 lookback=100, horizon=100, n_channels=1, custom_objs=[]):
        # type: (int, float, float, int, int, int, list) -> None
        self.n_hidden = units
        self.dropout_rate = dropout_rate
        self.lam = lam
        self.lookback = lookback
        self.horizon = horizon
        self.n_channels = n_channels
        self.filename = 'contseq2seq_{}_hidden_{}_dropout_{}_l2_lookback_{}_horizon_{}_channels_{}'.format(
                        self.n_hidden, self.dropout_rate, self.lam, self.lookback, self.horizon, self.n_channels)

        loss = 'mse'
        custom_objs = custom_objs

        lstm_spec = {'units': self.n_hidden,
                     'return_state': True,
                     'kernel_regularizer': l2(self.lam),
                     'recurrent_regularizer': l2(self.lam),
                     'dropout': self.dropout_rate,
                     'recurrent_dropout': self.dropout_rate}

        dense_spec = {'units': self.n_channels,
                      'activation': 'linear',
                      'kernel_regularizer': l2(self.lam)}

        infr_init_h = Input(shape=(self.n_hidden,))
        infr_init_C = Input(shape=(self.n_hidden,))

        encoder_input = Input(shape=(None, self.n_channels))
        decoder_input = Input(shape=(None, self.n_channels))
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

        decoder_dense = Dense(**dense_spec)
        decoded_sequence = decoder_dense(decoder_output)
        train_outputs = [decoded_sequence]
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
        cp_fname = 'cp_{}'.format(''.join([random.choice('0123456789ABCDEF') for _ in range(16)]))
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, verbose=1, mode='min'),
                     ModelCheckpoint(cp_fname, monitor='val_loss', mode='min',
                                     save_best_only=True,
                                     save_weights_only=True)]
        inputs = [train_frames[:, :self.lookback], train_frames[:, self.lookback:self.lookback + self.horizon]]
        outputs = [train_frames[:, self.lookback + 1:self.lookback + self.horizon + 1]]
        self.__sequence2sequence.fit(inputs, outputs, verbose=2, callbacks=callbacks, **kwargs)
        self.__sequence2sequence.load_weights(cp_fname)
        os.remove(cp_fname)

    def predict(self, inputs, predictive_horizon=100, mc_samples=100):
        samples = []
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

        return {'draws': np.stack(samples)}

    def save(self, folder, fname=None):
        save_obj = {'units': self.n_hidden,
                    'dropout_rate': self.dropout_rate,
                    'lam': self.lam,
                    'lookback': self.lookback,
                    'horizon': self.horizon,
                    'n_channels':self.n_channels}

        if fname is None:
            fname = ContinuousSeq2Seq.get_filename(save_obj)

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
        assert all([k in model_spec for k in ContinuousSeq2Seq.required_spec_keys])
        return 'seq2seq_{}_hidden_{}_dropout_{}_l2_lookback_{}_horizon_{}_channels_{}'.format(model_spec['units'],
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
        #print(weights_fname)
        assert weights_fname is not None, "Provide a valid weights filename to load model."

        model = ContinuousSeq2Seq(**spec)
        model.set_weights(weights_fname)

        return model

    @property
    def seed_length(self):
        return self.lookback + 1


class GPStrategy(ModelStrategy):
    """Implements the autoregressive Gaussian Process time series forecasting strategy."""
    id = 'argp'
    n_max_train = 10000

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
        if train_frames.shape[0] > self.n_max_train:
            print("Time series is too long!") #Training on first {} samples".format(self.n_max_train)

        self.model = gpy.models.GPRegression(train_frames[:self.n_max_train, :self.lookback, 0],  # TODO: attractor compatibility
                                             train_frames[:self.n_max_train, self.lookback:self.lookback+1, 0],
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


class GPFlowStrategy(ModelStrategy):
    """Implements the autoregressive Gaussian Process time series forecasting strategy and GPflow."""
    id = 'argpflow'
    n_max_train = 10000

    def __init__(self, ker, lookback=100, horizon=1, fname='tmp', n_channels=1, model=None, x_ranges=None):
        self.ker = ker + gpf.kernels.White(lookback)
        self.lookback = lookback
        self.horizon = horizon

        if 'argpflow' not in fname:
            self.fname = 'argpflow_{}'.format(fname)
        else:
            self.fname = fname

        self.model = model
        self.x_ranges = x_ranges

    @staticmethod
    def load(fname):
        svr = gpf.saver.Saver()
        model = svr.load(fname,
                         context=gpf.saver.SaverContext(autocompile=False))
        model.clear()
        model.compile()
        horizon = 1
        n_channels = model.Y.shape[-1]
        lookback = model.X.shape[-1] // n_channels
        x_ranges = [np.linspace(model.Y.value.min(axis=0), model.Y.value.max(axis=0), 1000)]
        this_fname = fname.split('/')[-1]
        return GPFlowStrategy(model.kern, lookback, horizon, this_fname, n_channels, model=model, x_ranges=x_ranges)

    def save(self, folder, fname='coreg_gp_tmp', overwrite=True):
        full_fname = '{}/{}'.format(folder, fname)
        if os.path.isfile(full_fname):
            if overwrite:
                os.remove(full_fname)
            else:
                print('Permission denied to duplicate file, please enable overwrite flag.')
                return -1

        svr = gpf.saver.Saver()
        svr.save(folder + fname, self.model)

    def fit(self, train_frames, restarts=1):
        if train_frames.shape[0] > self.n_max_train:
            print("Time series is too long!") #Training on first {} samples".format(self.n_max_train)

        #self.model = gpy.models.GPRegression(train_frames[:self.n_max_train, :self.lookback, 0],
        #                                     train_frames[:self.n_max_train, self.lookback:self.lookback+1, 0],
        #                                     self.ker)

        X = train_frames[:self.n_max_train, :self.lookback, 0]
        Y = train_frames[:self.n_max_train, self.lookback:self.lookback + 1, 0]
        self.x_ranges = [np.linspace(Y.min(), Y.max(), 1000)]

        self.model = gpf.models.GPR(X, Y, kern=self.ker)

        gpf.train.ScipyOptimizer().minimize(self.model)

    def predict(self, inputs, predictive_horizon=100, mc_samples=100):
        pred_inputs = inputs[:, -self.seed_length:, 0]
        assert pred_inputs.ndim == 2  # TODO: reshape for attractor compatibility
        assert self.model is not None

        pred_mean, pred_var = self.model.predict_y(pred_inputs)
        pred_sigma = np.sqrt(pred_var)

        samples = np.random.normal(loc=pred_mean, scale=pred_sigma, size=(mc_samples, 1))
        draws = np.hstack((np.repeat(pred_inputs, axis=0, repeats=mc_samples), samples))

        for i in range(predictive_horizon - 1):
            pred_mu, pred_var = self.model.predict_y(draws[:, -self.seed_length:])
            pred_sigma = np.sqrt(pred_var)#.clip(0.) # TODO: sigma greater than 0
            samples = np.random.normal(loc=pred_mu, scale=pred_sigma)
            draws = np.hstack((draws, samples))

        return {'draws': draws[:, -predictive_horizon:]}
    @staticmethod
    def get_filename(params):
        # type: (dict) -> str
        return 'argpflow_{}'.format(params.get('fname', 'tmp'))

    @property
    def seed_length(self):
        return self.lookback


class FactorisedGPStrategy(ModelStrategy):
    """Implements the autoregressive Gaussian Process time series forecasting strategy."""
    id = 'factorised_argp'
    n_max_train = 10000

    def __init__(self, ker, lookback=100, horizon=1, fname='tmp', n_channels=1):
        self.ker = ker + gpy.kern.White(lookback)
        self.lookback = lookback
        self.horizon = horizon
        self.fname = 'factorised_argp_{}'.format(fname)  # TODO: DEFINE KERNEL STR
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
        if train_frames.shape[0] > self.n_max_train:
            print("Time series is too long! Training on first {} samples".format(self.n_max_train))

        self.model = gpy.models.GPRegression(train_frames[:self.n_max_train, :self.lookback, 0],
                                             # TODO: attractor compatibility
                                             train_frames[:self.n_max_train, self.lookback:self.lookback + 1, 0],
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
            pred_sigma = np.sqrt(pred_var)  # .clip(0.) # TODO: sigma greater than 0
            samples = np.random.normal(loc=pred_mu, scale=pred_sigma)
            draws = np.hstack((draws, samples))

        return {'draws': draws[:, self.seed_length:]}  # TODO: attractor compatibility

    @staticmethod
    def get_filename(params):
        # type: (dict) -> str
        return '{}_{}'.format('factorised_argp', params.get('fname', 'tmp'))

    @property
    def seed_length(self):
        return self.lookback


class CoregionalisedGPStrategy(ModelStrategy):
    """Implements the autoregressive mixture-of-kernels Gaussian Process time series forecasting strategy."""
    id = 'coreg_argp'
    n_max_train = 4000
    n_max_iter = 15000
    default_p_inducing = 0.2

    def __init__(self, ker, lookback=100, horizon=1, fname='', n_channels=1, model=None, x_ranges=None):
        self.model = model

        if 'coreg_argp' not in fname:
            self.fname = 'coreg_argp_{}'.format(fname)
        else:
            self.fname = fname

        self.ker = ker
        self.lookback = lookback
        self.horizon = horizon
        self.n_channels = n_channels
        self.x_ranges = x_ranges

    @staticmethod
    def load(fname):
        svr = gpf.saver.Saver()
        model = svr.load(fname,
                         context=gpf.saver.SaverContext(autocompile=False))
        model.clear()
        model.compile()
        horizon = 1
        n_channels = model.Y.shape[-1]
        lookback = model.X.shape[-1] // n_channels
        x_mins = model.Y.value.min(axis=0)
        x_max = model.Y.value.max(axis=0)
        x_ranges = [np.linspace(xmi, xma, 1000) for xmi, xma in zip(x_mins, x_max)]
        this_fname = fname.split('/')[-1]
        return CoregionalisedGPStrategy(model.kern,
                                        lookback,
                                        horizon,
                                        this_fname,
                                        n_channels,
                                        model=model,
                                        x_ranges=x_ranges)

    def save(self, folder, fname='coreg_gp_tmp', overwrite=True):
        full_fname = '{}/{}'.format(folder, fname)
        if os.path.isfile(full_fname):
            if overwrite:
                os.remove(full_fname)
            else:
                print('Permission denied to duplicate file, please enable overwrite flag.')
                return -1

        svr = gpf.saver.Saver()
        svr.save(folder + fname, self.model)

    def fit(self, train_frames, feature_func=None, n_inducing=200, init_mu_var=None):
        if train_frames.shape[0] > self.n_max_train:
            print("Training on last {} samples".format(self.n_max_train))
            X, Y = to_contiguous(train_frames[-self.n_max_train:, :self.lookback]), \
                   to_contiguous(train_frames[-self.n_max_train:, -self.horizon:])
        else:
            X, Y = to_contiguous(train_frames[:, :self.lookback]), \
                   to_contiguous(train_frames[:, -self.horizon:])

        idx_inducing = np.arange(X.shape[0])
        np.random.shuffle(idx_inducing)
        feature = feature_func(X, n_inducing, self.n_channels)

        if init_mu_var is None:
            self.model = gpf.models.SVGP(X, Y, self.ker, gpf.likelihoods.Gaussian(), feat=feature)
        else:
            self.model = gpf.models.SVGP(X, Y,
                                         self.ker,
                                         gpf.likelihoods.Gaussian(),
                                         feat=feature,
                                         q_mu=init_mu_var['q_mu'],
                                         q_sqrt=init_mu_var['q_sqrt'])

        x_mins = Y.min(axis=0)
        x_max = Y.max(axis=0)
        self.x_ranges = [np.linspace(xmi, xma, 1000) for xmi, xma in zip(x_mins, x_max)]
        opt = gpf.train.ScipyOptimizer()
        opt.minimize(self.model, disp=True, maxiter=self.n_max_iter)

    def predict(self, inputs, predictive_horizon=100, mc_samples=100):
        # inputs must be (1, lookback, channels)
        if inputs.ndim < 3:
            pred_seed = inputs[np.newaxis, :].copy()
        else:
            pred_seed = inputs.copy()

        mu_0, sig_0 = self.model.predict_y(to_contiguous(pred_seed[0]).T)

        pred_samples = np.random.normal(mu_0, np.sqrt(sig_0), (mc_samples, 1, self.n_channels))
        pred_input = np.concatenate([np.repeat(pred_seed, repeats=mc_samples, axis=0),
                                     pred_samples],
                                    axis=1)

        for _ in range(predictive_horizon - 1):
            new_mu, new_sig = self.model.predict_y(to_contiguous(pred_input[:, -self.lookback:]))
            new_samples = np.concatenate([np.random.normal(new_mu[i],
                                                           np.sqrt(new_sig[i]),
                                                           (1, 1, self.n_channels))
                                          for i in range(mc_samples)], axis=0)
            pred_input = np.concatenate([pred_input, new_samples], axis=1)

        return {'draws': pred_input[:, self.seed_length:],
                'mvar_x_ranges': self.x_ranges}

        # pred_samples = self.model.predict_f_samples(to_contiguous(pred_seed), mc_samples)

        #pred_input = np.concatenate([np.repeat(pred_seed, repeats=mc_samples, axis=0),
        #                             mc_samples],
        #                            axis=1)

        #for _ in range(predictive_horizon - 1):
        #    new_samples = self.model.predict_f_samples(to_contiguous(pred_input[:, -self.lookback:]), 1)
        #    pred_input = np.concatenate([pred_input, new_samples[0, :, np.newaxis]], axis=1)

        #return {'draws': pred_input[:, self.seed_length:], 'mvar_x_ranges':self.x_ranges}

    @staticmethod
    def prepare_coreg_mode_dict(mode, base_ker, n_input_dim, n_channels, has_white=False):
        if mode =='shared_kernel_feats':
            if has_white:
                kernel = gpf.multioutput.kernels.SharedIndependentMok(base_ker(n_input_dim, ARD=True)
                                                                      + gpf.kernels.White(n_input_dim),
                                                                      output_dimensionality=n_channels)
            else:
                kernel = gpf.multioutput.kernels.SharedIndependentMok(base_ker(n_input_dim, ARD=True),
                                                                      output_dimensionality=n_channels)

            return {'kernel': kernel, 'feature_func': SparseGPFeatureSelector.shared_mof}
        elif mode == 'ind_kernel_shared_feats':
            if has_white:
                kernels = [base_ker(n_input_dim, ARD=True) + gpf.kernels.White(n_input_dim) for _ in range(n_channels)]
                kernel = gpf.multioutput.kernels.SeparateIndependentMok(kernels)
            else:
                kernels = [base_ker(n_input_dim, ARD=True) for _ in range(n_channels)]
                kernel = gpf.multioutput.kernels.SeparateIndependentMok(kernels)

            return {'kernel': kernel, 'feature_func': SparseGPFeatureSelector.shared_mof}
        elif mode == 'ind_kernel_feats':
            if has_white:
                kernels = [base_ker(n_input_dim, ARD=True) + gpf.kernels.White(n_input_dim) for _ in range(n_channels)]
                kernel = gpf.multioutput.kernels.SeparateIndependentMok(kernels)
            else:
                kernels = [base_ker(n_input_dim, ARD=True) for _ in range(n_channels)]
                kernel = gpf.multioutput.kernels.SeparateIndependentMok(kernels)
            return {'kernel': kernel, 'feature_func': SparseGPFeatureSelector.separate_mof}
        elif mode == 'mixed':
            if has_white:
                kernels = [base_ker(n_input_dim, ARD=True) + gpf.kernels.White(n_input_dim) for _ in range(n_channels)]
                kernel = gpf.multioutput.kernels.SeparateMixedMok(kernels, W=np.random.randn(n_channels, n_channels))
            else:
                kernels = [base_ker(n_input_dim, ARD=True) for _ in range(n_channels)]
                kernel = gpf.multioutput.kernels.SeparateMixedMok(kernels, W=np.random.randn(n_channels, n_channels))
            return {'kernel': kernel, 'feature_func': SparseGPFeatureSelector.mixed_kernel}

        print("Invalid mode \'{}\' provided to prepare_coreg_mode_dict".format(mode))
        return {}


    @staticmethod
    def get_filename(params):
        # type: (dict) -> str
        return '{}_{}'.format('coreg_argp', params.get('fname', 'tmp'))

    @property
    def seed_length(self):
        return self.lookback


class SparseGPFeatureSelector(object):
    @staticmethod
    def separate_mof(X, n_inducing, n_channels):
        N = X.shape[0]
        Zs = [X[np.random.permutation(N)[:n_inducing], ...].copy() for _ in range(n_channels)]
        # initialise as list inducing features
        feature_list = [gpf.features.InducingPoints(Z) for Z in Zs]
        # create multioutput features from Z
        return mf.SeparateIndependentMof(feature_list)

    @staticmethod
    def shared_mof(X, n_inducing, n_channels):
        idx_inducing = np.arange(X.shape[0])
        np.random.shuffle(idx_inducing)
        Z = X[idx_inducing[:n_inducing]].copy()
        return mf.SharedIndependentMof(gpf.features.InducingPoints(Z))

    @staticmethod
    def mixed_kernel(X, n_inducing, n_channels):
        idx_inducing = np.arange(X.shape[0])
        np.random.shuffle(idx_inducing)
        Z = X[idx_inducing[:n_inducing]].copy()
        return mf.MixedKernelSharedMof(gpf.features.InducingPoints(Z))
