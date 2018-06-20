import keras
from abc import ABCMeta, abstractmethod, abstractproperty
from keras import Model, Sequential, Input
from keras.layers import Dense, LSTM, Average, Bidirectional, Dropout
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
import pickle
from dataset import Dataset
import numpy as np
import random
import os


class ModelStrategy:
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
    required_spec_keys = ['ordinal_bins', 'units', 'dropout_rate', 'lambda', 'horizon', 'lookback']
    id = 'mordred'

    def __init__(self, model_spec):
        assert all([k in model_spec for k in self.required_spec_keys])
        self.n_bins = model_spec['ordinal_bins']
        self.n_hidden = model_spec['units']
        self.dropout_rate = model_spec['dropout_rate']
        self.lam = model_spec['lambda']
        self.lookback = model_spec['lookback']
        self.horizon = model_spec['horizon']
        self.filename = MordredStrategy.get_filename(model_spec)

        if 'custom_objs' not in model_spec: model_spec['custom_objs'] = []

        custom_objs = model_spec['custom_objs']

        neural_network_builder = KerasSequentialBuilder()

        lstm_spec = {'units': self.n_hidden,
                     'return_state': True,
                     'kernel_regularizer': self.lam,
                     'recurrent_regularizer': self.lam,
                     'dropout': self.dropout_rate,
                     'recurrent_dropout': self.dropout_rate}

        dense_spec = {'units': self.n_bins,
                      'activation': 'softmax',
                      'kernel_regularizer': self.lam}

        encoder_input = Input(shape=(None, self.n_bins))
        decoder_input = Input(shape=(None, self.n_bins))

        encoder_fwd = neural_network_builder.build_lstm(lstm_spec)
        lstm_spec['go_backwards'] = True
        encoder_bkwd = neural_network_builder.build_lstm(lstm_spec)
        _, h_fwd, C_fwd = encoder_fwd(encoder_input)
        _, h_bkwd, C_bkwd = encoder_bkwd(encoder_input)

        decoder_initial_states = [Average()([h_fwd, h_bkwd]), Average()([C_fwd, C_bkwd])]

        lstm_spec['return_sequences'] = True
        lstm_spec['go_backwards'] = False
        decoder_lstm = neural_network_builder.build_lstm(lstm_spec)
        decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=decoder_initial_states)
        decoder_dense = neural_network_builder.build_fully_connected(dense_spec)

        if self.dropout_rate > 0.:
            decoder_output = neural_network_builder.build_dropout(self.dropout_rate)(decoder_output)

        decoded_sequence = decoder_dense(decoder_output)
        self.__sequence2sequence = Model([encoder_input, decoder_input], decoded_sequence)

        loss = 'categorical_crossentropy'
        self.__sequence2sequence.compile(optimizer='nadam', loss=loss, metrics=[loss] + custom_objs)

        self.predict_stochastic = K.function([encoder_input, decoder_input, K.learning_phase()],
                                             [decoder_output])

        self.__encoder = Model(encoder_input, decoder_initial_states)
        self.predict_stochastic_encoder = K.function([encoder_input, K.learning_phase()],
                                                     decoder_initial_states)

        infr_init_h = Input(shape=(self.n_hidden,))
        infr_init_C = Input(shape=(self.n_hidden,))

        decoder_output, infr_h, infr_C = decoder_lstm(decoder_input, initial_state=[infr_init_h, infr_init_C])
        inferred_sequence = decoder_dense(decoder_output)

        self.__decoder = Model([decoder_input, infr_init_h, infr_init_C],
                               [inferred_sequence, infr_h, infr_C])
        self.predict_stochastic_decoder = K.function([decoder_input, infr_init_h, infr_init_C, K.learning_phase()],
                                                     [inferred_sequence, infr_h, infr_C])

    def fit(self, train_frames, **kwargs):
        # type: (np.ndarray) -> None
        cp_fname = 'cp_{}'.format(''.join([random.choice('0123456789ABCDEF') for _ in range(16)]))

        inputs = [train_frames[:, :self.lookback],
                  train_frames[:, self.lookback:self.lookback + self.horizon]]
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
        encoder_inputs = inputs[:, :self.lookback]

        for i_s in range(mc_samples):
            decoder_seed = inputs[:, self.lookback:self.lookback + 1]

            h, c = self.predict_stochastic_encoder([encoder_inputs, True])
            seq = []

            for t in range(predictive_horizon):
                decoder_seed, h, c = self.predict_stochastic_decoder([decoder_seed, h, c, True])
                seq += [decoder_seed]

            samples += [seq]

        posterior_mean = np.array(samples).mean(axis=0).squeeze()
        drawn_samples = [np.random.choice(self.n_bins, mc_samples, p=posterior_mean[t])
                         for t in range(predictive_horizon)]

        return {'ordinal_pdf': posterior_mean, 'draws': drawn_samples}

    def save(self, folder, fname=None):
        save_obj = {'ordinal_bins': self.n_bins,
                    'units': self.n_hidden,
                    'dropout_rate': self.dropout_rate,
                    'lambda': self.lam,
                    'lookback': self.lookback,
                    'horizon': self.horizon}

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
        return 'mordred_{}_bins_{}_hidden_{}_dropout_{}_l2_lookback_{}_horizon_{}'.format(model_spec['ordinal_bins'],
                                                                                          model_spec['units'],
                                                                                          model_spec['dropout_rate'],
                                                                                          model_spec['lambda'],
                                                                                          model_spec['lookback'],
                                                                                          model_spec['horizon'])

    @staticmethod
    def load(fname, custom_objs = None):
        with open(fname, 'r') as f:
            spec = pickle.load(f)

        if custom_objs is not None:
            spec['custom_objs'] = custom_objs

        model = MordredStrategy(spec)
        model.set_weights(spec['weights_fname'])

        return model

    @property
    def seed_length(self):
        return self.lookback + 1


class GPStrategy(ModelStrategy):
    def __init__(self):
        pass

    def load(fname): pass

    def save(self, fname): pass

    def fit(self, inputs, outputs, **kwargs): pass

    def predict(self, inputs, horizon=100): pass

    def draw_prediction_samples(self, inputs, n_samples=10): pass


class GPyKernelBuilder:
    def __init__(self, spec):
        pass


class KerasSequentialBuilder:
    def __init__(self):
        self.transformations = []
        self.model = None

    def build_lstm(self, spec):
        # type: (dict) -> keras.layers.Layer
        if 'kernel_regularizer' in spec and type(spec['kernel_regularizer']) is float:
            spec['kernel_regularizer'] = l2(spec['kernel_regularizer'])
        if 'recurrent_regularizer' in spec and type(spec['recurrent_regularizer']) is float:
            spec['recurrent_regularizer'] = l2(spec['recurrent_regularizer'])
        if 'bidirectional_merge_mode' in spec:
            bidirectional_merge_mode = spec['bidirectional_merge_mode']
            spec.pop('bidirectional_merge_mode')
        else:
            bidirectional_merge_mode = None

        new_layer = LSTM(**spec)

        if bidirectional_merge_mode is not None:
            new_layer = Bidirectional(new_layer, merge_mode=bidirectional_merge_mode)

        self.transformations += [new_layer]
        return new_layer

    def build_fully_connected(self, spec):
        # type: (dict) -> keras.layers.Layer
        if 'kernel_regularizer' in spec and type(spec['kernel_regularizer']) is float:
            spec['kernel_regularizer'] = l2(spec['kernel_regularizer'])

        new_layer = Dense(**spec)

        self.transformations += [new_layer]
        return new_layer

    def build_dropout(self, dropout_rate):
        new_layer = Dropout(dropout_rate)
        self.transformations += [new_layer]

        return new_layer

    def add_layer(self, layer_spec):
        # type: (str, dict) -> keras.layers.Layer
        layer_type = layer_spec['type']
        layer_spec.pop('type')
        if layer_type == 'lstm': return self.build_lstm(layer_spec)
        elif layer_type == 'dense': return self.build_fully_connected(layer_spec)
        elif layer_type == 'activation': return self.build_activation(layer_spec)
        else: 'Layer type {} not supported'.format(layer_type)

    def assemble_sequential_network(self, **compile_args):
        # type: () -> keras.Model
        if self.model is None: self.model = Sequential(self.transformations)
        self.model.compile(compile_args)
        return self.model
