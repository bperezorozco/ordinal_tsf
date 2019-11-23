from abc import ABCMeta, abstractmethod
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib.colors import ListedColormap
from ordinal_tsf.util import assert_sum_one, all_satisfy, frame_ts, is_univariate, frame_generator, frame_generator_list, gmm_marginal_pdf
import pickle
import numpy as np
import seaborn as sns
from scipy.stats import norm, multivariate_normal
from scipy.spatial.distance import euclidean
from functools import partial, reduce
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from fastdtw import fastdtw
import copy
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

class Dataset(object):
    """Handler for a time series dataset and its different representations."""
    train_ts = None
    val_ts = None
    test_ts = None

    def __init__(self, raw_ts, frame_length, p_train=0.7, p_val=0.15, p_test=0.15, preprocessing_steps=[]):
        assert assert_sum_one([p_train, p_val, p_test]) \
               and all_satisfy([p_train, p_val, p_test], lambda x: x > 0.), \
               "Please make sure p_train, p_val, p_test are positive and sum up to 1."

        assert raw_ts.ndim == 2 and raw_ts.shape[0], 'Please provide a univariate time series as input to Dataset.'

        self.optional_params = {}
        self.raw_ts = raw_ts
        self.frame_length = frame_length
        self.__raw_n_train = int(self.raw_ts.shape[0] * p_train)
        self.__raw_n_val = self.__raw_n_train + int(self.raw_ts.shape[0] * p_val)

        self.train_ts, self.val_ts, self.test_ts = self.raw_train_ts, self.raw_val_ts, self.raw_test_ts

        for step in preprocessing_steps:
            self.train_ts = step.apply(self.train_ts)
            self.val_ts = step.apply(self.val_ts)
            self.test_ts = step.apply(self.test_ts)

            self.optional_params.update(step.param_dict)

        if self.optional_params.get('frame_generator', False):
            self.train_frames = partial(frame_generator, ts=self.train_ts, frame_length=frame_length)
            self.val_frames = partial(frame_generator, ts=self.val_ts, frame_length=frame_length)
            self.test_frames = partial(frame_generator, ts=self.test_ts, frame_length=frame_length)
        else:
            self.train_frames = frame_ts(self.train_ts, frame_length)
            self.val_frames = frame_ts(self.val_ts, frame_length)
            self.test_frames = frame_ts(self.test_ts, frame_length)

    def save(self, fname):
        tmp = [self.train_frames, self.val_frames, self.test_frames]
        with open(fname, 'wb') as f:
            self.train_frames, self.val_frames, self.test_frames = None, None, None
            pickle.dump(self, f)
            self.train_frames, self.val_frames, self.test_frames = tmp

    @staticmethod
    def load(fname):
        with open(fname, 'r') as f:
            dataset = pickle.load(f)
            dataset.train_frames = frame_ts(dataset.train_ts, dataset.frame_length)
            dataset.val_frames = frame_ts(dataset.val_ts, dataset.frame_length)
            dataset.test_frames = frame_ts(dataset.test_ts, dataset.frame_length)
            return dataset

    @property
    def raw_train_ts(self):
        # type: (...) -> np.ndarray
        return self.raw_ts[:self.__raw_n_train]

    @property
    def raw_val_ts(self):
        # type: (...) -> np.ndarray
        return self.raw_ts[self.__raw_n_train:self.__raw_n_val]

    @property
    def raw_test_ts(self):
        # type: (...) -> np.ndarray
        return self.raw_ts[self.__raw_n_val:]

    def __str__(self):
        props = reduce(lambda x,y:x+y, ['{}:{}\n'.format(k,v) for k,v in self.optional_params.items()])
        return 'Dataset with properties:\n' + props

    def get_default_fname(self, id):
        """Dataset's default name based on its own preprocessing pipeline."""
        fname = id

        if 'white_noise_level' in self.optional_params:
            fname += '_sigma_{}'.format(self.optional_params['white_noise_level'])

        if 'zero_mean_unit_var' in self.optional_params:
            fname += '_standardised'

        if 'is_ordinal' in self.optional_params:
            fname += '_ordinal'

        if 'is_attractor' in self.optional_params:
            fname += '_attractor'

        return fname

    def apply_partial_preprocessing(self, mode, enabled_steps):
        """Queries a specific representation of the given dataset

        Applies a pipeline of preprocessing steps to obtain a dataset representation."""
        # type: (str, List[DatasetPreprocessingStep]) -> np.ndarray
        assert mode in ['train', 'test', 'val'], "Mode must be one of [train, val, test]"

        if mode == 'val':
            ts = self.raw_val_ts.copy()
        elif mode == 'test':
            ts = self.raw_test_ts.copy()
        else:
            ts = self.raw_train_ts.copy()

        for step in enabled_steps:
            ts = step.apply(ts)

        return ts


class TimeSeriesSetDataset(object):
    """This handler takes a list of time series and treats each of them as an individual subdataset"""
    def __init__(self, ts_list, frame_length, p_train=0.7, p_val=0.15, p_test=0.15,
                 preprocessing_steps=[],
                 multichannel_preprocessing_steps = [],
                 frame_gen_func=frame_generator_list):
        # type: (List[np.ndarray], int, float, float, float, List[DatasetPreprocessingStep]) -> TimeSeriesSetDataset

        assert [raw_ts.ndim == 2 and raw_ts.shape[0] > frame_length for raw_ts in ts_list], \
            'Please provide only univariate time series as input to Dataset.'

        self.raw_train_list = []
        self.raw_val_list = []
        self.raw_test_list = []
        self.frame_length = frame_length
        self.train_ts = []
        self.val_ts = []
        self.test_ts = []
        self.optional_params = {'is_list': True}
        self.optional_params_list = []
        self.preprocessing_steps_list = []
        self.frame_length = frame_length

        for ts in ts_list:
            n_train = int(p_train * ts.shape[0])
            n_train_val = int((p_train + p_val) * ts.shape[0])

            cur_train_ts = ts[:n_train]
            cur_val_ts = ts[n_train:n_train_val]
            cur_test_ts = ts[n_train_val:]

            self.raw_train_list += [cur_train_ts.copy()]
            self.raw_val_list += [cur_val_ts.copy()]
            self.raw_test_list += [cur_test_ts.copy()]

            current_optional_params = {}
            current_preproc_steps = []

            for step in preprocessing_steps:
                this_step = copy.deepcopy(step)

                cur_train_ts = this_step.apply(cur_train_ts)
                cur_val_ts = this_step.apply(cur_val_ts)
                cur_test_ts = this_step.apply(cur_test_ts)

                current_preproc_steps += [this_step]
                current_optional_params.update(this_step.param_dict)

            self.optional_params_list += [current_optional_params]
            self.train_ts += [cur_train_ts]
            self.val_ts += [cur_val_ts]
            self.test_ts += [cur_test_ts]
            self.preprocessing_steps_list += [current_preproc_steps]

        self.train_frames = partial(frame_gen_func, ts_list=self.train_ts, frame_length=frame_length)
        self.val_frames = partial(frame_gen_func, ts_list=self.val_ts, frame_length=frame_length)
        self.test_frames = partial(frame_gen_func, ts_list=self.test_ts, frame_length=frame_length)

    def apply_partial_preprocessing(self, mode, enabled_steps):
        """Queries a specific representation of the given dataset

        Applies a pipeline of preprocessing steps to obtain a dataset representation."""
        # type: (str, List[DatasetPreprocessingStep]) -> np.ndarray
        assert mode in ['train', 'test', 'val'], "Mode must be one of [train, val, test]"

        if mode == 'val':
            ts = self.raw_val_list.copy()
        elif mode == 'test':
            ts = self.raw_test_list.copy()
        else:
            ts = self.raw_train_list.copy()

        for step in enabled_steps:
            ts = step.apply(ts)

        return ts


class DatasetPreprocessingStep(object):
    """Provides a common interface for the individual transformations of the dataset preprocessing pipeline

    Attributes:
        is_fitted (bool): All preprocessing steps have to be fitted to the input time series
        param_dict (dict): Communication protocol from the step's attributes that must be known by the caller
    """
    __metaclass__ = ABCMeta
    is_fitted = False
    param_dict = {}

    @abstractmethod
    def apply(self, ts):
        """Common interface to perform a transformation from a raw time series to its new representation"""
        # type: (np.ndarray) -> np.ndarray
        pass


class WhiteCorrupter(DatasetPreprocessingStep):
    """Adds white noise to a time series

    Args:
        sigma (float): noise standard deviation
    """
    def __init__(self, sigma=1e-3):
        self.noise_level = sigma

    def apply(self, ts):
        self.param_dict['white_noise_level'] = self.noise_level
        return ts + np.random.normal(scale=self.noise_level, size=ts.shape)


class FrameGenerator(DatasetPreprocessingStep):
    """Ensures the time series framer will be a generator

    Args:
        sigma (float): noise standard deviation
    """
    def __init__(self):
        self.param_dict['frame_generator'] = True

    def apply(self, ts):
        return ts


class Standardiser(DatasetPreprocessingStep):
    """Makes a time series zero-mean, unit-variance"""
    def __init__(self):
        self.mean = None
        self.std = None

    def apply(self, ts):
        if not self.is_fitted: self.fit(ts)

        return (ts - self.mean) / self.std

    def fit(self, ts):
        self.mean = np.nanmean(ts)
        self.std = np.nanstd(ts)
        self.param_dict = {'ts_mean': self.mean,
                           'ts_std': self.std,
                           'zero_mean_unit_var':True}
        self.is_fitted = True


class MultivarStandardiser(DatasetPreprocessingStep):
    """Makes a time series zero-mean, unit-variance"""
    def __init__(self):
        self.mean = None
        self.std = None

    def apply(self, ts):
        if isinstance(ts,(list,)):
            ts = np.stack(ts, axis=-1)

        if not self.is_fitted: self.fit(ts)

        return (ts - self.mean) / self.std

    def fit(self, ts):
        if isinstance(ts,(list,)):
            ts = np.stack(ts, axis=-1)

        self.mean = np.nanmean(ts, axis=0)
        self.std = np.nanstd(ts, axis=0)
        self.param_dict = {'ts_mean': self.mean,
                           'ts_std': self.std,
                           'zero_mean_unit_var':True}

        self.is_fitted = True


class Quantiser(DatasetPreprocessingStep):
    """Computes ordinal bins and allocates each observation in the time series."""
    def __init__(self, n_bins=None, delta=1e-3, frame_generator=True):
        self.n_bins = n_bins
        self.delta = delta
        self.bins = None
        self.frame_generator = frame_generator

    def apply(self, ts):
        if not self.is_fitted:
            self.fit(ts)
        assert is_univariate(ts), 'Only univariate time series can be quantised. Current shape: {}'.format(ts.shape)

        if (ts.max() > (self.bins[-1]+self.delta)) or (ts.min() < (self.bins[0]-self.delta)):
            print("WARNING: You are trying to quantise a time series that has observations outside the quantisation "
                  "range. BE CAREFUL as This may lead to inaccurate results.")

        na_mask = np.isnan(ts)

        out = np.zeros((ts.shape[0], self.n_bins))
        digits = np.searchsorted(self.bins[:-1], ts.squeeze())

        for i, i_d in enumerate(digits):
            if na_mask[i]:
                out[i, :] = np.nan
            else:
                out[i, i_d] = 1.

        return out

    def fit(self, ts):
        ts_max = np.nanmax(ts)
        ts_min = np.nanmin(ts)

        if self.n_bins is None:
            self.n_bins = self.__find_n_bins(ts)

        self.bins = np.linspace(ts_min, ts_max, self.n_bins)

        self.param_dict = {'bins': self.bins,
                           'bin_delta':self.bins[1]-self.bins[0],
                           'is_ordinal': True,
                           'frame_generator': self.frame_generator}

        self.is_fitted = True

    def __find_n_bins(self, ts):
        # type: (np.ndarray) -> int
        MAX_ALLOWED = 300
        MIN_ALLOWED = 10
        n_bins = np.unique(ts.squeeze()).shape[0]

        if n_bins < MAX_ALLOWED and n_bins > MIN_ALLOWED:
            return n_bins

        ts_max = np.nanmax(ts)
        ts_min = np.nanmin(ts)
        n_bins = int((ts_max - ts_min) / self.delta)
        n_bins = max(min(MAX_ALLOWED, n_bins), MIN_ALLOWED)

        return n_bins


class QuantiserArray(DatasetPreprocessingStep):
    """Computes ordinal bins and allocates each observation in the time series."""

    def __init__(self, n_bins=None, delta=1e-3, frame_generator=True):
        self.n_bins = n_bins
        self.delta = delta
        self.bins = None
        self.frame_generator = frame_generator
        self.quantisers = []

    def apply(self, ts):
        if not self.is_fitted:
            self.fit(ts)

        return np.stack([q.apply(ts[:, i_q:i_q + 1]) for i_q, q in enumerate(self.quantisers)], axis=-1)

    def fit(self, ts):
        for i_q in range(ts.shape[-1]):
            q = Quantiser(n_bins=self.n_bins, delta=self.delta, frame_generator=self.frame_generator)
            q.fit(ts[:, i_q:i_q + 1])
            self.quantisers += [q]

        self.n_bins = [q.n_bins for q in self.quantisers]
        self.bins = [q.bins for q in self.quantisers]

        self.param_dict = {'is_ordinal': True,
                           'frame_generator': self.frame_generator,
                           'is_array': True,
                           'n_channels': ts.shape[-1],
                           'bins': self.bins
                           }

        self.is_fitted = True


class KMeansQuantiser(DatasetPreprocessingStep):
    """Quantises a time series using the KMeans method"""
    def __init__(self, n_clusters=150, n_init=5):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.model = None

    def fit(self, ts):
        self.model = KMeans(n_clusters=self.n_clusters, n_init=self.n_init).fit(ts)
        self.param_dict = {'centroids': self.model.cluster_centers_,
                           'n_centroids': self.n_clusters,
                           'frame_generator': True}
        self.is_fitted = True

    def apply(self, ts):
        if not self.is_fitted:
            self.fit(ts)

        cluster_ids = self.model.predict(ts)
        out = np.zeros((ts.shape[0], self.n_clusters))

        for i, i_d in enumerate(cluster_ids):
            out[i, i_d] = 1.

        return out

    def apply_decoder(self, weights, f_decoder=None):
        if f_decoder is None:
            return self.replace_with_centroids(weights, self.param_dict['centroids'])

        return f_decoder(weights, self.param_dict['centroids'])

    @staticmethod
    def replace_with_centroids(draw, centroids):
        return np.stack([centroids[k] for k in draw])

    @staticmethod
    def replace_with_weighted_mean(weights, centroids):
        return weights.dot(centroids)

    @staticmethod
    def replace_with_weighted_mode(weights, centroids):
        modes = weights.argmax(axis=-1)
        return np.stack([centroids[i] for i in modes])


class GMMQuantiser(DatasetPreprocessingStep):
    """Quantises a time series using the Variational GMM method"""

    def __init__(self, n_clusters=150, n_init=5, weight_concentration_prior=500):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.model = None
        self.wcp = weight_concentration_prior

    def fit(self, ts):
        self.model = GaussianMixture(n_components=self.n_clusters, max_iter=200, n_init=self.n_init).fit(ts)

        self.param_dict = {'centroids': self.model.means_,
                           'covariances': self.model.covariances_,
                           'n_centroids': self.n_clusters,
                           'frame_generator': True}
        self.is_fitted = True

    def apply(self, ts):
        if not self.is_fitted:
            self.fit(ts)

        cluster_ids = self.model.predict(ts)
        out = np.zeros((ts.shape[0], self.n_clusters))

        for i, i_d in enumerate(cluster_ids):
            out[i, i_d] = 1.

        return out

    def compute_bin_proba(self, samples):
        return self.model.predict_proba(samples)

    def apply_decoder(self, weights, f_decoder=None):
        if f_decoder is None:
            return self.replace_with_centroids(weights, self.param_dict['centroids'])

        return f_decoder(weights, self.param_dict['centroids'])

    @staticmethod
    def replace_with_centroids(draw, centroids):
        return np.stack([centroids[k] for k in draw])

    @staticmethod
    def replace_with_weighted_mean(weights, centroids):
        return weights.dot(centroids)


class VBGMMQuantiser(DatasetPreprocessingStep):
    """Quantises a time series using the GMM method"""

    def __init__(self, n_clusters=150, n_init=5, weight_concentration_prior=500):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.model = None
        self.wcp = weight_concentration_prior

    def fit(self, ts):
        self.model = BayesianGaussianMixture(n_components=self.n_clusters, max_iter=200, n_init=self.n_init,
                                             weight_concentration_prior_type='dirichlet_distribution',
                                             weight_concentration_prior=self.wcp).fit(ts)

        self.param_dict = {'centroids': self.model.means_,
                           'covariances': self.model.covariances_,
                           'n_centroids': self.n_clusters,
                           'frame_generator': True}
        self.is_fitted = True

    def apply(self, ts):
        if not self.is_fitted:
            self.fit(ts)

        cluster_ids = self.model.predict(ts)
        out = np.zeros((ts.shape[0], self.n_clusters))

        for i, i_d in enumerate(cluster_ids):
            out[i, i_d] = 1.

        return out

    def apply_decoder(self, weights, f_decoder=None):
        if f_decoder is None:
            return self.replace_with_centroids(weights, self.param_dict['centroids'])

        return f_decoder(weights, self.param_dict['centroids'])

    @staticmethod
    def replace_with_centroids(draw, centroids):
        return np.stack([centroids[k] for k in draw])

    @staticmethod
    def replace_with_weighted_mean(weights, centroids):
        return weights.dot(centroids)


class AttractorStacker(DatasetPreprocessingStep):
    """Stacks a time series with lagged representations of itself, in an attractor-like fashion."""
    def __init__(self, lag):
        self.lag = lag

    def apply(self, ts):
        self.is_fitted = True
        self.param_dict = {'attractor_lag': self.lag,
                           'n_channels': 3,
                           'is_attractor': True}
        return np.stack((ts[:-2*self.lag], ts[self.lag:-self.lag], ts[2*self.lag:]), axis=-1)


class Selector(DatasetPreprocessingStep):
    """Extracts a subsequence of length ``horizon`` from index ``start``"""
    def __init__(self, start, horizon):
        self.start = start
        self.end = start + horizon

    def apply(self, ts):
        self.is_fitted = True
        return ts[self.start:self.end]


class Prediction(object):
    """Provides a common interface for the output predictions of different forecasting strategies    """
    __metaclass__ = ABCMeta
    type = 'deterministic'

    @abstractmethod
    def mse(self, ground_truth): pass

    @abstractmethod
    def nll(self, ground_truth): pass

    @staticmethod
    def get_mase_norm_constant(tr_ts, m):
        n = tr_ts.shape[0]
        return np.abs(tr_ts[m:] - tr_ts[:-m]).sum() / (n - m)


class StateDensity(object):
    """Provides a common interface for the output predictions of different forecasting strategies    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def pdf(self, x): pass


class MultivariateOrdinalPrediction(Prediction):
    type = 'multi_ordinal'

    def __init__(self, quant, ordinal_pdf, draws, n_x=250):
        self.ordinal_pdf = ordinal_pdf
        self.quant = quant
        self.mean = quant.apply_decoder(ordinal_pdf, quant.replace_with_weighted_mean)
        bins = quant.param_dict['centroids']
        self.draws = quant.apply_decoder(draws, quant.replace_with_centroids)
        self.n_channels = bins.shape[-1]
        self.channel_ranges = []
        self.channel_densities = []
        self.gmm_marginals = []

        for i_channel in range(self.n_channels):
            this_range = np.linspace(self.draws[:, :, i_channel].min(),
                                     self.draws[:, :, i_channel].max(),
                                     n_x)[:, np.newaxis]
            this_delta = this_range[1] - this_range[0]

            this_channel_marginals = [norm(loc=quant.model.means_[k, i_channel],
                                           scale=np.sqrt(quant.model.covariances_[k, i_channel, i_channel]))
                                      for k in range(ordinal_pdf.shape[1])]

            this_channel_density = gmm_marginal_pdf(this_range, this_channel_marginals, ordinal_pdf, this_delta)

            self.channel_ranges += [this_range]
            self.channel_densities += [this_channel_density]
            self.gmm_marginals += [this_channel_marginals]

    def plot_channel_like(self, plt, y_true):
        fig, axes = plt.subplots(self.n_channels)

        for i_channel in range(self.n_channels):
            im = axes[i_channel].imshow(self.channel_densities[i_channel].T, origin='lower',
                                        extent=[0, self.channel_densities[i_channel].shape[0],
                                                self.channel_ranges[i_channel][0],
                                                self.channel_ranges[i_channel][-1]],
                                        aspect='auto', cmap='Blues')
            axes[i_channel].plot(y_true[:, i_channel], color='xkcd:green')
            axes[i_channel].plot(self.mean[:, i_channel], color='xkcd:orange')
            axes[i_channel].plot(self.get_ordinal_quantile(self.channel_densities[i_channel],
                                                           self.channel_ranges[i_channel], 0.5),
                                 color='xkcd:crimson')
            axes[i_channel].plot(self.get_ordinal_quantile(self.channel_densities[i_channel],
                                                           self.channel_ranges[i_channel], 0.025),
                                 color='xkcd:crimson')
            axes[i_channel].plot(self.get_ordinal_quantile(self.channel_densities[i_channel],
                                                           self.channel_ranges[i_channel], 0.975),
                                 color='xkcd:crimson')
            fig.colorbar(im, ax=axes[i_channel])
            # plt.colorbar()

    def plot_channel_cdf(self, plt, y_true):
        fig, axes = plt.subplots(self.n_channels)
        c_pal = sns.color_palette('Blues', n_colors=125).as_hex()
        my_cmap = ListedColormap(c_pal + c_pal[::-1][1:])

        for i_channel in range(self.n_channels):
            im = axes[i_channel].imshow(self.channel_densities[i_channel].cumsum(axis=-1).T, origin='lower',
                                        extent=[0, self.channel_densities[i_channel].shape[0],
                                                self.channel_ranges[i_channel][0],
                                                self.channel_ranges[i_channel][-1]],
                                        aspect='auto', cmap=my_cmap)
            axes[i_channel].plot(y_true[:, i_channel], color='xkcd:green')
            axes[i_channel].plot(self.mean[:, i_channel], color='xkcd:orange')
            axes[i_channel].plot(self.get_ordinal_quantile(self.channel_densities[i_channel],
                                                           self.channel_ranges[i_channel], 0.5),
                                 color='xkcd:crimson')
            axes[i_channel].plot(self.get_ordinal_quantile(self.channel_densities[i_channel],
                                                           self.channel_ranges[i_channel], 0.025),
                                 color='xkcd:crimson')
            axes[i_channel].plot(self.get_ordinal_quantile(self.channel_densities[i_channel],
                                                           self.channel_ranges[i_channel], 0.975),
                                 color='xkcd:crimson')
            fig.colorbar(im, ax=axes[i_channel])

    def plot_decoded(self, plt, y_true):
        fig = plt.figure()

        if y_true.shape[1] == 3:
            ax = fig.gca(projection='3d')
            ax.plot(y_true[:, 0], y_true[:, 1], y_true[:, 2], '.', color='xkcd:crimson')
            ax.plot(self.mean[:, 0], self.mean[:, 1], self.mean[:, 2], '.', color='xkcd:blue')
        elif y_true.shape[1] == 2:
            ax = fig.gca()
            ax.plot(y_true[:, 0], y_true[:, 1], '.', color='xkcd:crimson')
            ax.plot(self.mean[:, 0], self.mean[:, 1], '.', color='xkcd:blue')
        else:
            print('Incorrect number of channels')

    def plot_channels(self, plt, y_true):
        fig, axes = plt.subplots(y_true.shape[-1])

        for i_ax, ax in enumerate(axes):
            ax.plot(self.mean[:, i_ax], color='xkcd:blue')
            ax.plot(y_true[:, i_ax], color='xkcd:crimson')

    def get_ordinal_quantile(self, pdf, x_range, alpha):
        cdf = pdf.cumsum(axis=-1)
        quantile = np.array([x_range[j] for j in (cdf >= alpha).argmax(axis=-1)])
        quantile[cdf[:, -1] < alpha] = x_range[-1]

        return quantile

    def rmse_mean(self, ground_truth):
        return np.sqrt(mean_squared_error(ground_truth, self.mean.squeeze()))

    def mse(self):
        pass

    def nll(self, y_true):
        #bin_proba = self.quant.compute_bin_proba(y_true)

        bin_proba = np.stack([multivariate_normal.pdf(y_true,
                                                 self.quant.model.means_[k_mix],
                                                 self.quant.model.covariances_[k_mix])
                         for k_mix in range(self.quant.model.means_.shape[0])], axis=1)

        p_ground_truth = (bin_proba * self.ordinal_pdf).sum(axis=-1)
        return (-np.log(p_ground_truth)).sum()


# This is used when you have independent models for each time series channel and then want to
# integrate into a single prediction
class PredictionList(Prediction):
    def __init__(self, predictions):
        self.n_ar_channels = len(predictions)
        self.predictions = predictions

    def mse(self, ground_truth): return 0.

    def nll(self, ground_truth): return 0.

    def rmse_mean(self, ground_truth):
        # ground_truth \in (timesteps, channels)
        mse = np.array([pred.rmse_mean(ground_truth[i_pred]) for i_pred, pred in enumerate(self.predictions)])
        return mse.mean()

    def plot_channel_like(self, plt, ground_truth):
        fig, axes = plt.subplots(self.n_ar_channels)

        if self.n_ar_channels == 1:
            axes = [axes]

        for i_pred, pred in enumerate(self.predictions):
            pred.plot_empirical(axes[i_pred], ground_truth[i_pred])

    def plot_median_2std(self, plt, ground_truth):
        fig, axes = plt.subplots(self.n_ar_channels)

        if self.n_ar_channels == 1:
            axes = [axes]

        for i_pred, pred in enumerate(self.predictions):
            pred.plot_median_2std(axes[i_pred], ground_truth[i_pred])

    def ordinal_marginal_nll(self, ordinal_ground_truth):
        return np.array([pred.nll(ordinal_ground_truth[i]) for i, pred in enumerate(self.predictions)])


class OrdinalPrediction(Prediction):
    """Encapsulates a sequential ordinal predictive posterior distribution.

       This implements the strategy to compute metrics and plots where the predictive distribution is assumed to be
       ordinal/categorical at every timestep.

       Args:
           ordinal_pdf (np.ndarray): The ordinal output of the forecasting model
           draws (np.ndarray): The draws obtained from the forecasting model
           bins (np.ndarray): The bins used the decode the sample trajectory draws

       Attributes:
           ordinal_pdf (np.ndarray): The ordinal output of the forecasting model
           draws (np.ndarray): The draws obtained from the forecasting model
           bins (np.ndarray): The bins used the decode the sample trajectory draws
           delta (float): Bin width used to reinterpret ordinal pdf as a piecewise uniform pdf
       """

    type = 'ordinal'

    def __init__(self, ordinal_pdf, draws, bins):
        self.ordinal_pdf = ordinal_pdf
        self.draws = np.array([[bins[j] for j in draw] for draw in draws])
        self.bins = bins
        self.delta = self.bins[1] - self.bins[0]

    def mse(self, ground_truth):
        """Computes MSE between two real-valued time series"""
        # type: (np.ndarray) -> np.float
        return np.mean([mean_squared_error(ground_truth, p) for p in self.draws])

    def smape_mean(self, ground_truth):
        this_mean = self.ordinal_pdf.dot(self.bins).squeeze()
        k = ground_truth.shape[0]
        y_true = ground_truth.squeeze()
        smape_vector = np.abs(y_true - this_mean) / (np.abs(y_true) + np.abs(this_mean))

        return smape_vector.sum() * (2. / k)

    def smape_quantile(self, ground_truth, alpha=0.5):
        k = ground_truth.shape[0]
        median = self.get_quantile(alpha).squeeze()
        y_true = ground_truth.squeeze()
        smape_vector = np.abs(y_true - median) / (np.abs(y_true) + np.abs(median))

        return smape_vector.sum() * (2. / k)

    def mase_mean(self, ground_truth, mase_norm_constant):
        k = ground_truth.shape[0]
        #mase_norm_constant = self.get_mase_norm_constant(ground_truth, 1)
        this_mean = self.ordinal_pdf.dot(self.bins).squeeze()
        y_true = ground_truth.squeeze()
        mase_vector = np.abs(y_true - this_mean).sum() / k

        return mase_vector / mase_norm_constant

    def mase_quantile(self, ground_truth, mase_norm_constant, alpha=0.5):
        k = ground_truth.shape[0]
        #mase_norm_constant = self.get_mase_norm_constant(ground_truth, 1)
        median = self.get_quantile(alpha).squeeze()
        y_true = ground_truth.squeeze()
        mase_vector = np.abs(y_true - median).sum() / k

        return mase_vector / mase_norm_constant

    def rmse_quantile(self, ground_truth, alpha=0.5):
        return np.sqrt(mean_squared_error(ground_truth, self.get_quantile(alpha).squeeze()))

    def rmse_mean(self, ground_truth):
        return np.sqrt(mean_squared_error(ground_truth, self.ordinal_pdf.dot(self.bins).squeeze()))

    def mse_mean(self, ground_truth):
        return mean_squared_error(ground_truth, self.ordinal_pdf.dot(self.bins).squeeze())

    def quantile_mse(self, ground_truth, alpha=0.5):
        return mean_squared_error(ground_truth, self.get_quantile(alpha).squeeze())

    def mse_and_std(self, ground_truth):
        """Computes MSE +- STD between two real-valued time series"""
        # type: (np.ndarray) -> np.float
        all_mse = [mean_squared_error(ground_truth, prediction) for prediction in self.draws]
        return np.mean(all_mse), np.std(all_mse)

    def median_dtw_distance(self, ground_truth):
        pred_median = self.get_quantile(0.5)
        dist, path = fastdtw(pred_median, ground_truth, dist=euclidean)

        return dist

    def median_attractor_distance(self, ground_truth):
        pred_median = self.get_quantile(0.5)
        stacker = AttractorStacker(10)
        pred_median_att = stacker.apply(pred_median).squeeze()
        ground_truth_att = stacker.apply(ground_truth).squeeze()

        d = pairwise_distances(pred_median_att, ground_truth_att)
        return d.min(axis=0).sum()

    def nll(self, binned_ground_truth):
        """Computes NLL of drawing a time series from a piecewise uniform sequential prediction"""
        # type: (np.ndarray) -> np.float
        p_ground_truth = (self.ordinal_pdf * binned_ground_truth / self.delta).max(axis=-1)
        neg_log_p_ground_truth = -np.log(p_ground_truth)
        return neg_log_p_ground_truth.sum()

    def qq_dist(self, ordinal_ground_truth, up_to=1000):
        qq_x = np.arange(0.01, 1., 0.01)
        y_pred_idx = (self.ordinal_pdf[:up_to] * ordinal_ground_truth[:up_to]).argmax(axis=-1)
        cdf_truth = np.array([self.ordinal_pdf[t, :idx].sum() for t, idx in enumerate(y_pred_idx)])
        qq_ordinal = np.array([(cdf_truth <= alpha).mean() for alpha in qq_x])

        return mean_squared_error(qq_x, qq_ordinal)

    def cum_nll(self, binned_ground_truth):
        """Computes integral of NLL(t) of drawing a time series from a piecewise uniform sequential prediction"""
        # type: (np.ndarray) -> np.float
        p_ground_truth = (self.ordinal_pdf * binned_ground_truth / self.delta).max(axis=-1)
        neg_log_p_ground_truth = -np.log(p_ground_truth)
        return neg_log_p_ground_truth.cumsum().sum()

    def get_quantile(self, alpha):
        """Computes \alpha-quantiles given the object's ordinal pdf"""
        # type: (float) -> np.ndarray
        cdf = self.ordinal_pdf.cumsum(axis=-1)
        return np.array([self.bins[j] for j in (cdf >= alpha).argmax(axis=-1)])

    def plot_median_2std(self, plt, ground_truth):
        """Plots a probabilistic forecast's median and 2.5, 97.5 quantiles alongside the corresponding ground truth"""
        quantile_025 = self.get_quantile(0.025)
        quantile_975 = self.get_quantile(0.975)
        quantile_median = self.get_quantile(0.5)

        plt.plot(quantile_025, 'xkcd:orange')
        plt.plot(quantile_975, 'xkcd:orange')
        plt.plot(quantile_median, 'xkcd:maroon')
        plt.plot(ground_truth, 'xkcd:olive')
        plt.legend(['Quantile 0.025', 'Quantile 0.975', 'Median', 'True'])

    def plot_mean_2std(self, plt, ground_truth):
        """Plots a probabilistic forecast's median and 2.5, 97.5 quantiles alongside the corresponding ground truth"""
        quantile_025 = self.get_quantile(0.025)
        quantile_975 = self.get_quantile(0.975)
        pred_mean = self.ordinal_pdf.dot(self.bins)

        plt.plot(quantile_025, 'xkcd:orange')
        plt.plot(quantile_975, 'xkcd:orange')
        plt.plot(pred_mean, 'xkcd:maroon')
        plt.plot(ground_truth, 'xkcd:olive')
        plt.legend(['Quantile 0.025', 'Quantile 0.975', 'Mean', 'True'])

    def plot_like(self, plt, ground_truth=None):
        """Plots the full ordinal pdf as a heatmap"""
        if ground_truth is not None:
            plt.plot(ground_truth, 'xkcd:orange')

        plt.imshow(self.ordinal_pdf.T, origin='lower',
                   extent=[0, self.ordinal_pdf.shape[0], self.bins.min(), self.bins.max()],
                   aspect='auto', cmap='Blues')
        plt.title('Predictive likelihood')
        plt.colorbar()

    def plot_cum_nll(self, plt, binned_ground_truth):
        """Plots the full ordinal pdf as a heatmap"""
        p_ground_truth = (self.ordinal_pdf * binned_ground_truth / self.delta).max(axis=-1)
        neg_log_p_ground_truth = -np.log(p_ground_truth)
        cum_nll = neg_log_p_ground_truth.cumsum()

        plt.plot(cum_nll)
        plt.title('Cumulative negative log likelihood')

    def plot_log_like(self, plt, ground_truth=None):
        """Plots the full log ordinal pdf as a heatmap"""
        if ground_truth is not None:
            plt.plot(ground_truth, 'xkcd:orange')

        plt.imshow(np.ma.log(self.ordinal_pdf.T).data, origin='lower',
                   extent=[0, self.ordinal_pdf.shape[0], self.bins.min(), self.bins.max()],
                   aspect='auto', cmap='Blues')
        plt.title('Predictive log likelihood')
        plt.colorbar()

    def plot_draws_quantiles(self, plt, ground_truth):
        """Plots a probabilistic forecast's median and 2.5, 97.5 quantiles alongside the corresponding ground truth"""
        quantile_025 = self.get_quantile(0.025)
        quantile_975 = self.get_quantile(0.975)
        quantile_median = self.get_quantile(0.5)

        [plt.plot(x, color='xkcd:blue', alpha=0.1) for x in self.draws.squeeze()]
        plt.plot(quantile_025, 'xkcd:orange')
        plt.plot(quantile_975, 'xkcd:orange')
        plt.plot(quantile_median, 'xkcd:maroon')
        plt.plot(ground_truth, 'xkcd:olive')
        plt.legend(['Quantile 0.025', 'Quantile 0.975', 'Median', 'True'], bbox_to_anchor=(1., 1.))

    def plot_empirical(self, plt, ground_truth):
        c_pal = sns.color_palette('Blues', n_colors=150).as_hex()
        my_cmap = ListedColormap(c_pal + c_pal[::-1][1:])
        quantile_025 = self.get_quantile(0.025)
        quantile_975 = self.get_quantile(0.975)

        plt.plot(quantile_025, 'xkcd:azure')
        plt.plot(quantile_975, 'xkcd:azure')
        plt.plot(ground_truth, 'xkcd:coral')
        plt.imshow(self.ordinal_pdf.cumsum(axis=-1).T, origin='lower',
                   extent=[0, ground_truth.shape[0], self.bins.min(), self.bins.max()],
                   aspect='auto', cmap=my_cmap)
        #plt.title('Empirical distribution function')
        #plt.colorbar()

    def plot_qq(self, plt, ordinal_ground_truth, up_to=1000, col='xkcd:blue'):
        qq_x = np.arange(0.01, 1., 0.01)
        y_pred_idx = (self.ordinal_pdf[:up_to] * ordinal_ground_truth[:up_to]).argmax(axis=-1)
        cdf_truth = np.array([self.ordinal_pdf[t, :idx].sum() for t, idx in enumerate(y_pred_idx)])
        qq_ordinal = np.array([(cdf_truth <= alpha).mean() for alpha in qq_x])
        plt.plot(qq_x, qq_ordinal, col)
        plt.plot(qq_x, qq_x, '--', color='xkcd:green')
        plt.legend(['Ordinal prediction', 'Ideal'])
        #plt.title('Uncertainty calibration plot for ordinal prediction')

    def plot_median_dtw_alignment(self, plt, ground_truth):
        pred_median = self.get_quantile(0.5)
        dist, path = fastdtw(pred_median, ground_truth, dist=euclidean)
        plt.plot(np.array([pred_median[j] for i, j in path]))
        plt.plot(np.array([ground_truth[i] for i, j in path]))

    @staticmethod
    def compatibility(old_pred):
        new_pred = OrdinalPrediction(old_pred.ordinal_pdf, [], old_pred.bins)
        new_pred.draws = old_pred.draws
        return new_pred


# This is used when you have a shared LSTM layer and only individual fully connected layers
# for each output channel
class OrdinalArrayPrediction(Prediction):
    type = 'ordinal_array'

    def __init__(self, ordinal_pdf, draws, bins, vbgmm_max_components=5):
        self.predictions = []
        self.n_channels = ordinal_pdf.shape[-1]
        self.vbgmm_components = vbgmm_max_components
        for i in range(ordinal_pdf.shape[-1]):
            self.predictions += [OrdinalPrediction(ordinal_pdf[:, :, i],
                                                   draws[:, :, i],
                                                   bins[i])]
        self.draws = np.stack([pred.draws for pred in self.predictions], axis=-1)
        self.vbgmm = [BayesianGaussianMixture(vbgmm_max_components, n_init=3, max_iter=200).fit(self.draws[:, t])
                      for t in range(self.draws.shape[1])]
        x_mins = [b[0] for b in bins]
        x_max = [b[-1] for b in bins]
        self.x_ranges = np.stack([np.linspace(xmi, xma, 1000) for xmi, xma in zip(x_mins, x_max)], axis=-1)

    def get_quantile(self, alpha):
        """Computes \alpha-quantiles given the object's posterior mean and standard deviation"""
        # type: (float) -> np.ndarray
        all_quantiles = []
        for i_ch in range(self.n_channels):
            this_quantile = [self.x_ranges[q, i_ch]
                              for q in (self.all_ch_cdf[:, :, i_ch] >= alpha).argmax(axis=-1)] # shape: (n_ts, n_ts_range, n_channels)
            this_quantile = np.array(this_quantile)
            msk = (self.all_ch_cdf[:, -1, i_ch] < alpha)
            this_quantile[msk] = self.x_ranges[-1, i_ch]
            all_quantiles += [this_quantile]

        return np.stack(all_quantiles, axis=-1)

    def mse(self, ground_truth):
        """Computes MSE between two real-valued time series"""
        # type: (np.ndarray) -> np.float
        return np.mean([pred.mse(ground_truth[:, :, i]) for i, pred in enumerate(self.predictions)])

    def smape_mean(self, ground_truth):
        return -1.

    def rmse_quantile(self, ground_truth, alpha=0.5):
        return np.mean([pred.rmse_quantile(ground_truth[:, i], alpha) for i, pred in enumerate(self.predictions)])

    def rmse_mean(self, ground_truth):
        return np.mean([pred.rmse_mean(ground_truth[:, i]) for i, pred in enumerate(self.predictions)])

    def nll(self, ground_truth):
        """Computes NLL of drawing a time series from a piecewise uniform sequential prediction"""
        # type: (np.ndarray) -> np.float
        return -np.sum([self.vbgmm[t].score(ground_truth[t:t+1]) for t in range(self.draws.shape[1])])
        #return np.sum([pred.nll(binned_ground_truth[:, :, i])
        #               for i, pred in enumerate(self.predictions)])

    def plot_median_2std(self, plt, ground_truth):
        fig, axes = plt.subplots(self.n_channels)
        for i_pred, pred in enumerate(self.predictions):
            pred.plot_median_2std(axes[i_pred], ground_truth[:, i_pred])

    def plot_decoded(self, plt, ground_truth):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #[ax.plot(*d.T, linestyle='', marker='.', color='xkcd:blue', alpha=0.01) for d in self.draws]
        ax.plot(*self.draws.mean(axis=0).T, linestyle='', marker='.', color='xkcd:blue')
        ax.plot(*ground_truth.T, linestyle='-', marker='.', color='xkcd:crimson')

    def plot_channel_like(self, plt, ground_truth):
        fig, axes = plt.subplots(self.n_channels)

        for i_pred, pred in enumerate(self.predictions):
            pred.plot_empirical(axes[i_pred], ground_truth[:, i_pred])

    def plot_draws_quantiles(self, plt, ground_truth):
        fig, axes = plt.subplots(self.n_channels)

        for i_pred, pred in enumerate(self.predictions):
            pred.plot_draws_quantiles(axes[i_pred], ground_truth[:, i_pred])

    def factorised_ordinal_joint_nll(self, ordinal_ground_truth):
        return np.sum([pred.nll(ordinal_ground_truth[:, :, i]) for i, pred in enumerate(self.predictions)])

    def ordinal_marginal_nll(self, ordinal_ground_truth):
        return np.array([pred.nll(ordinal_ground_truth[:, :, i]) for i, pred in enumerate(self.predictions)])

    def vbgmm_joint_nll(self, ground_truth):
        return -np.sum([self.vbgmm[t].score(ground_truth[t:t + 1]) for t in range(self.draws.shape[1])])

    def vbgmm_marginal_nll(self, ground_truth):
        all_ch_like = []
        for i_ch in range(self.n_channels):
            ch_like = []
            for t in range(ground_truth.shape[0]):
                cur_ch_like = 0.
                for k_mix in range(self.vbgmm[t].weights_.shape[0]):
                    cur_ch_like += self.vbgmm[t].weights_[k_mix] * norm.pdf(ground_truth[t:t + 1, i_ch],
                                                                           loc=self.vbgmm[t].means_[k_mix, i_ch],
                                                                           scale=np.sqrt(self.vbgmm[t].covariances_[k_mix,
                                                                                                                i_ch,
                                                                                                                i_ch]))
                ch_like += [cur_ch_like]
            all_ch_like += [-np.log(ch_like).sum()]

        return all_ch_like


class StatePrediction(Prediction):
    type = 'state'


class GaussianPrediction(Prediction):
    """Encapsulates a sequential Gaussian predictive posterior distribution.

    This implements the strategy to compute metrics and plots where the predictive distribution assumed to be
    Gaussian at every timestep.

    Args:
        draws (np.ndarray): The draws obtained from the forecasting model

    Attributes:
        posterior_mean (np.ndarray): Monte Carlo approximation of the posterior predictive mean
        posterior_std (np.ndarray): Monte Carlo approximation of the posterior predictive standard deviation
    """

    type = 'gaussian'

    def __init__(self, draws, raw_pred=None):
        if raw_pred is None:
            self.posterior_mean = draws.mean(axis=0)
            self.posterior_std = draws.std(axis=0)
            self.draws = draws
        else:
            self.posterior_mean = raw_pred['posterior_mean'].squeeze()
            self.posterior_std = raw_pred['posterior_std'].squeeze()
            self.draws = np.stack([np.random.normal(self.posterior_mean[t],
                                                    self.posterior_std[t],
                                                    size = 100) for t in range(self.posterior_mean.shape[0])], axis=1)

    def mse(self, ground_truth):
        """Computes MSE between two real-valued time series"""
        # type: (np.ndarray) -> np.float
        return np.mean([mean_squared_error(ground_truth, p) for p in self.draws])

    def rmse_quantile(self, ground_truth, alpha=0.5):
        return np.sqrt(mean_squared_error(ground_truth, self.get_quantile(alpha).squeeze()))

    def rmse_mean(self, ground_truth):
        return np.sqrt(mean_squared_error(ground_truth, self.posterior_mean))

    def mse_mean(self, ground_truth):
        return mean_squared_error(ground_truth, self.posterior_mean)

    def quantile_mse(self, ground_truth, alpha=0.5):
        return mean_squared_error(ground_truth, self.get_quantile(alpha).squeeze())

    def median_dtw_distance(self, ground_truth):
        pred_median = self.get_quantile(0.5)
        dist, path = fastdtw(pred_median, ground_truth, dist=euclidean)

        return dist

    def median_attractor_distance(self, ground_truth):
        pred_median = self.get_quantile(0.5)
        stacker = AttractorStacker(10)
        pred_median_att = stacker.apply(pred_median).squeeze()
        ground_truth_att = stacker.apply(ground_truth).squeeze()

        d = pairwise_distances(pred_median_att, ground_truth_att)
        return d.min(axis=0).sum()

    def smape_mean(self, ground_truth):
        this_mean = self.posterior_mean.squeeze()
        y_true = ground_truth.squeeze()
        k = ground_truth.shape[0]
        smape_vector = np.abs(y_true - this_mean) / (np.abs(y_true) + np.abs(this_mean))

        return smape_vector.sum() * (2. / k)

    def smape_quantile(self, ground_truth, alpha=0.5):
        k = ground_truth.shape[0]
        y_true = ground_truth.squeeze()
        median = self.get_quantile(alpha).squeeze()
        smape_vector = np.abs(y_true - median) / (np.abs(y_true) + np.abs(median))

        return smape_vector.sum() * (2. / k)

    def mase_mean(self, ground_truth, mase_norm_constant):
        k = ground_truth.shape[0]
        y_true = ground_truth.squeeze()
        #mase_norm_constant = self.get_mase_norm_constant(ground_truth, 1)
        this_mean = self.posterior_mean.squeeze()
        mase_vector = np.abs(y_true - this_mean).sum() / k

        return mase_vector / mase_norm_constant

    def mase_quantile(self, ground_truth, mase_norm_constant, alpha=0.5):
        k = ground_truth.shape[0]
        y_true = ground_truth.squeeze()
        #mase_norm_constant = self.get_mase_norm_constant(ground_truth, 1)
        median = self.get_quantile(alpha).squeeze()
        mase_vector = np.abs(y_true - median).sum() / k

        return mase_vector / mase_norm_constant

    def nll(self, ground_truth):
        """Computes NLL of drawing a time series from a GP sequential prediction"""
        # type: (np.ndarray) -> np.float
        horizon = self.posterior_mean.shape[0]
        likelihood = np.array([norm(loc=self.posterior_mean[i], scale=self.posterior_std[i]).pdf(ground_truth[i])
                               for i in range(horizon)])

        log_like = -np.log(likelihood)
        if np.isinf(log_like).any():
            likelihood += 1e-12
            likelihood /= likelihood.sum()
            log_like = -np.log(likelihood)

        nll = log_like.sum()
        #print 'NLL: {}'.format(nll)
        return nll

    def qq_dist(self, ground_truth, up_to=1000):
        qq_x = np.arange(0.01, 1., 0.01)
        qq_gp = [np.less_equal(ground_truth.squeeze()[:up_to], self.get_quantile(a)[:up_to]).mean() for a in qq_x]
        return mean_squared_error(qq_x, qq_gp)

    def get_quantile(self, alpha):
        """Computes \alpha-quantiles given the object's posterior mean and standard deviation"""
        # type: (float) -> np.ndarray
        return np.array([norm.ppf(alpha, mu, sigma) for mu, sigma in zip(self.posterior_mean, self.posterior_std)])

    def cum_nll(self, ground_truth):
        """Computes compulative NLL of drawing a time series from a GP sequential prediction"""
        # type: (np.ndarray) -> np.float
        horizon = self.posterior_mean.shape[0]
        likelihood = np.array([norm(loc=self.posterior_mean[i], scale=self.posterior_std[i]).pdf(ground_truth[i])
                               for i in range(horizon)])

        nll = -np.log(likelihood).cumsum().sum()
        #print 'Cum NLL: {}'.format(nll)
        return nll

    def plot_cum_nll(self, plt, ground_truth):
        """Computes compulative NLL of drawing a time series from a GP sequential prediction"""
        # type: (np.ndarray) -> np.float
        horizon = self.posterior_mean.shape[0]
        likelihood = np.array([norm(loc=self.posterior_mean[i], scale=self.posterior_std[i]).pdf(ground_truth[i])
                               for i in range(horizon)])

        nll = -np.log(likelihood).cumsum()
        plt.plot(nll)
        plt.title('Cumulative negative log likelihood')

    def plot_median_2std(self, plt, ground_truth):
        """Plots a probabilistic forecast's median and 2.5, 97.5 quantiles alongside the corresponding ground truth"""
        quantile_median = self.posterior_mean
        quantile_025 = quantile_median - 2 * self.posterior_std
        quantile_975 = quantile_median + 2 * self.posterior_std

        plt.plot(quantile_025, 'xkcd:orange')
        plt.plot(quantile_975, 'xkcd:orange')
        plt.plot(quantile_median, 'xkcd:maroon')
        plt.plot(ground_truth, 'xkcd:olive')
        plt.legend(['Quantile 0.025', 'Quantile 0.975', 'Median', 'True'])

    def plot_draws_quantiles(self, plt, ground_truth):
        """Plots a probabilistic forecast's median and 2.5, 97.5 quantiles alongside the corresponding ground truth"""
        quantile_median = self.posterior_mean
        quantile_025 = quantile_median - 2 * self.posterior_std
        quantile_975 = quantile_median + 2 * self.posterior_std

        [plt.plot(x, color='xkcd:blue', alpha=0.1) for x in self.draws.squeeze()]
        plt.plot(quantile_025, 'xkcd:orange')
        plt.plot(quantile_975, 'xkcd:orange')
        plt.plot(quantile_median, 'xkcd:maroon')
        plt.plot(ground_truth, 'xkcd:olive')
        plt.legend(['Quantile 0.025', 'Quantile 0.975', 'Median', 'True'])

    def plot_empirical(self, plt, ground_truth):
        x_min = ground_truth.min()
        x_max = ground_truth.max()
        x = np.linspace(x_min, x_max, 300)

        cdf = np.stack([norm.cdf(x, mu, sigma) for mu, sigma in zip(self.posterior_mean, self.posterior_std)], axis=0)

        quantile_025 = self.get_quantile(0.025)
        quantile_975 = self.get_quantile(0.975)

        plt.plot(quantile_025, 'xkcd:azure')
        plt.plot(quantile_975, 'xkcd:azure')

        c_pal = sns.color_palette('Blues', n_colors=150).as_hex()
        my_cmap = ListedColormap(c_pal + c_pal[::-1][1:])

        plt.plot(ground_truth, 'xkcd:coral')
        plt.imshow(cdf.T, origin='lower',
                   extent=[0, cdf.shape[0], x_min, x_max],
                   aspect='auto', cmap=my_cmap)
        #plt.title('Empirical distribution function')
        #plt.colorbar()

    def plot_qq(self, plt, ground_truth, up_to=1000, col='xkcd:blue'):
        qq_x = np.arange(0.01, 1., 0.01)
        qq_gp = [np.less_equal(ground_truth.squeeze()[:up_to], self.get_quantile(a)[:up_to]).mean() for a in qq_x]
        plt.plot(qq_x, qq_gp, color=col)
        plt.plot(qq_x, qq_x, '--', color='xkcd:green')
        plt.legend(['Continuous prediction', 'Ideal'])
        #plt.title('Uncertainty calibration plot for continuous prediction')

    @staticmethod
    def compatibility(old_pred):
        return GaussianPrediction(old_pred.draws)


class GaussianMixturePrediction(Prediction):
    type = 'gmm'

    def __init__(self, draws, n_components, vbgmms=None):
        draw_length = draws.shape[1]
        self.n_components = n_components
        self.draws = draws
        overall_x_min = None
        overall_x_max = None

        if vbgmms is None:
            self.vbgmms = []
            for t in range(draw_length):
                self.vbgmms += [BayesianGaussianMixture(self.n_components, n_init=3, max_iter=200).fit(self.draws[:, t, np.newaxis])]
        else:
            self.vbgmms = vbgmms

        for vbgmm in self.vbgmms:
            x_min = (vbgmm.means_.squeeze() - 3. * np.sqrt(vbgmm.covariances_).squeeze()).min()
            x_max = (vbgmm.means_.squeeze() + 3. * np.sqrt(vbgmm.covariances_).squeeze()).max()

            if overall_x_min is None or x_min < overall_x_min:
                overall_x_min = x_min

            if overall_x_max is None or x_max > overall_x_max:
                overall_x_max = x_max

        x = np.linspace(overall_x_min, overall_x_max, 300)
        self.ts_range = x
        self.cdf = self.eval_cdf(x)

    def mse(self, ground_truth):
        """Computes MSE between two real-valued time series"""
        # type: (np.ndarray) -> np.float
        return np.mean([mean_squared_error(ground_truth, p) for p in self.draws])

    def mse_mean(self, ground_truth):
        return mean_squared_error(ground_truth, self.draws.mean(axis=0))

    def rmse_quantile(self, ground_truth, alpha=0.5):
        return np.sqrt(mean_squared_error(ground_truth, self.get_quantile(alpha).squeeze()))

    def rmse_mean(self, ground_truth):
        return np.sqrt(mean_squared_error(ground_truth, self.draws.mean(axis=0).squeeze()))

    def median_dtw_distance(self, ground_truth):
        pred_median = self.get_quantile(0.5)
        dist, path = fastdtw(pred_median, ground_truth, dist=euclidean)

        return dist

    def median_attractor_distance(self, ground_truth):
        pred_median = self.get_quantile(0.5)
        stacker = AttractorStacker(10)
        pred_median_att = stacker.apply(pred_median).squeeze()
        ground_truth_att = stacker.apply(ground_truth).squeeze()

        d = pairwise_distances(pred_median_att, ground_truth_att)
        return d.min(axis=0).sum()

    def smape_mean(self, ground_truth):
        this_mean = self.draws.mean(axis=0).squeeze()
        y_true = ground_truth.squeeze()
        k = ground_truth.shape[0]
        smape_vector = np.abs(y_true - this_mean) / (np.abs(y_true) + np.abs(this_mean))

        return smape_vector.sum() * (2. / k)

    def smape_quantile(self, ground_truth, alpha=0.5):
        k = ground_truth.shape[0]
        y_true = ground_truth.squeeze()
        median = self.get_quantile(alpha).squeeze()
        smape_vector = np.abs(y_true - median) / (np.abs(y_true) + np.abs(median))

        return smape_vector.sum() * (2. / k)

    def mase_mean(self, ground_truth, mase_norm_constant):
        k = ground_truth.shape[0]
        y_true = ground_truth.squeeze()
        #mase_norm_constant = self.get_mase_norm_constant(ground_truth, 1)
        this_mean = self.draws.mean(axis=0).squeeze()
        mase_vector = np.abs(y_true - this_mean).sum() / k

        return mase_vector / mase_norm_constant

    def mase_quantile(self, ground_truth, mase_norm_constant, alpha=0.5):
        k = ground_truth.shape[0]
        #mase_norm_constant = self.get_mase_norm_constant(ground_truth, 1)
        y_true = ground_truth.squeeze()
        median = self.get_quantile(alpha).squeeze()
        mase_vector = np.abs(y_true - median).sum() / k

        return mase_vector / mase_norm_constant

    def quantile_mse(self, ground_truth, alpha=0.5):
        return mean_squared_error(ground_truth, self.get_quantile(alpha).squeeze())

    def qq_dist(self, ground_truth, up_to=1000):
        qq_x = np.arange(0.01, 1., 0.01)
        qq_gp = [np.less_equal(ground_truth.squeeze()[:up_to], self.get_quantile(a)[:up_to]).mean() for a in qq_x]
        return mean_squared_error(qq_x, qq_gp)

    def nll(self, ground_truth):
        """Computes NLL of drawing a time series from a GP sequential prediction"""
        # type: (np.ndarray) -> np.float
        horizon = len(self.vbgmms)
        likelihood = []

        for t in range(horizon):
            vbgmm = self.vbgmms[t]
            p = 0.
            for pi, mu, sigma_sq in zip(vbgmm.weights_.squeeze(), vbgmm.means_.squeeze(), vbgmm.covariances_.squeeze()):
                sigma = np.sqrt(sigma_sq)
                p += pi * norm.pdf(ground_truth[t], mu, sigma)

            likelihood += [p]

        likelihood = np.array(likelihood)
        nll = -np.log(likelihood).sum()
        #print 'NLL: {}'.format(nll)
        return nll

    def get_quantile(self, alpha):
        """Computes \alpha-quantiles given the object's posterior mean and standard deviation"""
        # type: (float) -> np.ndarray
        quantile = np.array([self.ts_range[j] for j in (self.cdf >= alpha).argmax(axis=-1)])

        return np.array([self.ts_range[j] for j in (self.cdf >= alpha).argmax(axis=-1)])

    def cum_nll(self, ground_truth):
        """Computes compulative NLL of drawing a time series from a GP sequential prediction"""
        # type: (np.ndarray) -> np.float
        horizon = len(self.vbgmms)
        likelihood = []

        for t in range(horizon):
            vbgmm = self.vbgmms[t]
            p = 0.
            for pi, mu, sigma_sq in zip(vbgmm.weights_.squeeze(), vbgmm.means_.squeeze(), vbgmm.covariances_.squeeze()):
                sigma = np.sqrt(sigma_sq)
                p += pi * norm.pdf(ground_truth[t], mu, sigma)

            likelihood += [p]

        likelihood = np.array(likelihood)

        nll = -np.log(likelihood).cumsum().sum()
        #print 'Cum NLL: {}'.format(nll)
        return nll

    def plot_cum_nll(self, plt, ground_truth):
        """Computes compulative NLL of drawing a time series from a GP sequential prediction"""
        # type: (np.ndarray) -> np.float
        horizon = len(self.vbgmms)
        likelihood = []

        for t in range(horizon):
            vbgmm = self.vbgmms[t]
            p = 0.
            for pi, mu, sigma_sq in zip(vbgmm.weights_.squeeze(), vbgmm.means_.squeeze(), vbgmm.covariances_.squeeze()):
                sigma = np.sqrt(sigma_sq)
                p += pi * norm.pdf(ground_truth[t], mu, sigma)

            likelihood += [p]

        likelihood = np.array(likelihood)

        nll = -np.log(likelihood).cumsum()
        plt.plot(nll)
        plt.title('Cumulative negative log likelihood')

    def plot_median_2std(self, plt, ground_truth):
        """Plots a probabilistic forecast's median and 2.5, 97.5 quantiles alongside the corresponding ground truth"""
        quantile_025 = self.get_quantile(0.025)
        quantile_975 = self.get_quantile(0.975)
        quantile_median = self.get_quantile(0.5)

        plt.plot(quantile_025, 'xkcd:orange')
        plt.plot(quantile_975, 'xkcd:orange')
        plt.plot(quantile_median, 'xkcd:maroon')
        plt.plot(ground_truth, 'xkcd:olive')
        plt.legend(['Quantile 0.025', 'Quantile 0.975', 'Median', 'True'])

    def eval_cdf(self, x):
        cdf = []

        for vbgmm in self.vbgmms:
            P = 0.
            for pi, mu, sigma_sq in zip(vbgmm.weights_.squeeze(), vbgmm.means_.squeeze(), vbgmm.covariances_.squeeze()):
                sigma = np.sqrt(sigma_sq)
                P += pi * norm.cdf(x, mu, sigma)

            cdf += [P]

        return np.stack(cdf, axis=0)

    def plot_draws_quantiles(self, plt, ground_truth):
        """Plots a probabilistic forecast's median and 2.5, 97.5 quantiles alongside the corresponding ground truth"""
        quantile_025 = self.get_quantile(0.025)
        quantile_975 = self.get_quantile(0.975)
        quantile_median = self.get_quantile(0.5)

        [plt.plot(x, color='xkcd:blue', alpha=0.1) for x in self.draws.squeeze()]
        plt.plot(quantile_025, 'xkcd:orange')
        plt.plot(quantile_975, 'xkcd:orange')
        plt.plot(quantile_median, 'xkcd:maroon')
        plt.plot(ground_truth, 'xkcd:olive')
        plt.legend(['Quantile 0.025', 'Quantile 0.975', 'Median', 'True'])

    def plot_empirical(self, plt, ground_truth):
        c_pal = sns.color_palette('Blues', n_colors=150).as_hex()
        my_cmap = ListedColormap(c_pal + c_pal[::-1][1:])

        quantile_025 = self.get_quantile(0.025)
        quantile_975 = self.get_quantile(0.975)

        plt.plot(quantile_025, 'xkcd:azure')
        plt.plot(quantile_975, 'xkcd:azure')

        plt.plot(ground_truth, 'xkcd:coral')
        plt.imshow(self.cdf.T, origin='lower',
                   extent=[0, self.cdf.shape[0], self.ts_range.min(), self.ts_range.max()],
                   aspect='auto', cmap=my_cmap)
        plt.title('Empirical distribution function')
        #plt.colorbar()

    def plot_qq(self, plt, ground_truth, up_to=1000, col='xkcd:blue'):
        qq_x = np.arange(0.01, 1., 0.01)
        qq_gp = [np.less_equal(ground_truth.squeeze()[:up_to], self.get_quantile(a)[:up_to]).mean() for a in qq_x]
        plt.plot(qq_x, qq_gp, color=col)
        plt.plot(qq_x, qq_x, '--', color='xkcd:green')
        plt.legend(['Continuous prediction', 'Ideal'])
        #plt.title('Uncertainty calibration plot for continuous prediction')

    def plot_median_dtw_alignment(self, plt, ground_truth):
        pred_median = self.get_quantile(0.5)
        dist, path = fastdtw(pred_median, ground_truth, dist=euclidean)

        plt.plot(np.array([pred_median[j] for i, j in path]))
        plt.plot(np.array([ground_truth[i] for i, j in path]))

    @staticmethod
    def compatibility_univar_gaussian(old_pred, n_components):
        return GaussianMixturePrediction(old_pred.draws, n_components)

    @staticmethod
    def compatibility(old_pred):
        return GaussianMixturePrediction(old_pred.draws, old_pred.n_components, old_pred.vbgmms)


class MultivarVBGMMPrediction(Prediction):
    type = 'multi_vbgmm'

    def __init__(self, draws, x_ranges, vbgmms=None, n_components=5):
        self.draws = draws
        self.n_channels = draws.shape[-1]
        self.predictive_horizon = draws.shape[1]

        if vbgmms is not None:
            self.vbgmm = vbgmms
        else:
            self.vbgmm = [BayesianGaussianMixture(n_components, n_init=5, max_iter=200).fit(draws[:, t])
                        for t in range(self.predictive_horizon)]
        self.x_ranges = np.stack(x_ranges, axis=-1)
        self.all_ch_like = self.eval_marginal_like(self.x_ranges)
        self.all_ch_cdf = self.eval_marginal_cdf(self.x_ranges) # shape: (n_ts, n_ts_range, n_channels)
        self.pred_mean = self.draws.mean(axis=0)

    def get_quantile(self, alpha):
        """Computes \alpha-quantiles given the object's posterior mean and standard deviation"""
        # type: (float) -> np.ndarray
        all_quantiles = []
        for i_ch in range(self.n_channels):
            this_quantile = [self.x_ranges[q, i_ch]
                              for q in (self.all_ch_cdf[:, :, i_ch] >= alpha).argmax(axis=-1)] # shape: (n_ts, n_ts_range, n_channels)
            this_quantile = np.array(this_quantile)
            msk = (self.all_ch_cdf[:, -1, i_ch] < alpha)
            this_quantile[msk] = self.x_ranges[-1, i_ch]
            all_quantiles += [this_quantile]

        return np.stack(all_quantiles, axis=-1)

    def plot_decoded(self, plt, ground_truth):
        if self.n_channels == 3:
            ax = plt.figure(figsize=(12, 12)).gca(projection='3d')
        elif self.n_channels == 2:
            ax = plt.figure(figsize=(12, 12)).gca()
        else:
            print("plot_decoded only available for time series with n_channels < 4. Provided "
                  "time series has {} channels".format(ground_truth.shape[-1]))
            return

        [ax.plot(*p.T, '.', color='xkcd:blue', alpha=0.01) for p in self.draws]
        ax.plot(*ground_truth.T, color='xkcd:orange', label='Ground truth', linewidth=3.0)
        plt.legend(loc=7, prop={'size': 14})
        plt.title('Sample predictions and ground truth', fontsize=24)

    def plot_channel_cdf(self, plt, ground_truth):
        fig, axes = plt.subplots(self.n_channels, figsize=(15, 10))
        fig.suptitle('Cumulative predictive posterior', fontsize=24)
        c_pal = sns.color_palette('Blues', n_colors=150).as_hex()
        my_cmap = ListedColormap(c_pal + c_pal[::-1][1:])
        upper_quant = self.get_quantile(0.975)
        lower_quant = self.get_quantile(0.025)

        for i_ch in range(self.n_channels):
            im=axes[i_ch].imshow(self.all_ch_cdf.T[i_ch],
                              origin='lower',
                              extent=[0, self.predictive_horizon,
                                      self.x_ranges[0, i_ch], self.x_ranges[-1, i_ch]],
                              aspect='auto', cmap=my_cmap)
            axes[i_ch].plot(lower_quant[:, i_ch], 'xkcd:azure', label='Quantiles')
            axes[i_ch].plot(upper_quant[:, i_ch], 'xkcd:azure')
            axes[i_ch].plot(ground_truth[:, i_ch], color='xkcd:orange', label='Ground truth')
            axes[i_ch].legend(loc=1, prop={'size': 14})
            divider = make_axes_locatable(axes[i_ch])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax)

    def plot_channel_like(self, plt, ground_truth):
        fig, axes = plt.subplots(self.n_channels, figsize=(15, 10))

        upper_quant = self.get_quantile(0.975)
        lower_quant = self.get_quantile(0.025)
        fig.suptitle('Predictive posterior', fontsize=24)

        for i_ch in range(self.n_channels):
            im = axes[i_ch].imshow(self.all_ch_like.T[i_ch],
                              origin='lower',
                              extent=[0, self.predictive_horizon,
                                      self.x_ranges[0, i_ch], self.x_ranges[-1, i_ch]],
                              aspect='auto', cmap='Blues', norm=LogNorm(vmin=0.001, vmax=1))

            axes[i_ch].plot(lower_quant[:, i_ch], 'xkcd:azure', label='Quantiles')
            axes[i_ch].plot(upper_quant[:, i_ch], 'xkcd:azure')
            axes[i_ch].plot(ground_truth[:, i_ch], color='xkcd:orange', label='Ground truth')
            axes[i_ch].legend(loc=1, prop={'size': 14})
            divider = make_axes_locatable(axes[i_ch])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax)

    def plot_median_2std(self, plt, ground_truth, with_draws=True):
        upper_quant = self.get_quantile(0.975)
        lower_quant = self.get_quantile(0.025)
        pred_median = self.get_quantile(0.5)

        fig, axes = plt.subplots(self.n_channels, figsize=(15, 10))
        fig.suptitle('Predictive median and quantiles 2.5 and 97.5', fontsize=24)
        for i_ch in range(self.n_channels):
            axes[i_ch].plot(lower_quant[:, i_ch], 'xkcd:azure', label='Quantiles')
            axes[i_ch].plot(upper_quant[:, i_ch], 'xkcd:azure')
            axes[i_ch].plot(pred_median[:, i_ch], 'xkcd:azure')

            if with_draws:
                [axes[i_ch].plot(d[:, i_ch], alpha=0.05, color='xkcd:blue') for d in self.draws]

            axes[i_ch].plot(ground_truth[:, i_ch], color='xkcd:orange', label='Ground truth')
            axes[i_ch].legend(loc=1, prop={'size': 14})

    def plot_mean_2std(self, plt, ground_truth, with_draws=True):
        upper_quant = self.get_quantile(0.975)
        lower_quant = self.get_quantile(0.025)

        fig, axes = plt.subplots(self.n_channels, figsize=(15, 10))
        fig.suptitle('Predictive mean and quantiles 2.5 and 97.5', fontsize=24)
        for i_ch in range(self.n_channels):
            axes[i_ch].plot(lower_quant[:, i_ch], 'xkcd:azure', label='Quantiles')
            axes[i_ch].plot(upper_quant[:, i_ch], 'xkcd:azure')
            axes[i_ch].plot(self.pred_mean[:, i_ch], 'xkcd:azure')

            if with_draws:
                [axes[i_ch].plot(d[:, i_ch], alpha=0.05, color='xkcd:blue') for d in self.draws]

            axes[i_ch].plot(ground_truth[:, i_ch], color='xkcd:orange', label='Ground truth')
            axes[i_ch].legend(loc=1, prop={'size': 14})

    def rmse_mean(self, ground_truth):
        return np.sqrt(mean_squared_error(ground_truth, self.pred_mean))

    def rmse_quantile(self, ground_truth, alpha=0.5):
        return np.sqrt(mean_squared_error(ground_truth, self.get_quantile(alpha)))

    def vbgmm_joint_nll(self, ground_truth):
        return -np.sum([self.vbgmm[t].score(ground_truth[t:t + 1]) for t in range(self.draws.shape[1])])

    def vbgmm_marginal_nll(self, ground_truth):
        all_ch_like = []
        n_mix = self.vbgmm[0].weights_.shape[0]

        for i_ch in range(self.n_channels):
            ch_like = []
            for t in range(self.draws.shape[1]):
                cur_ch_like = 0.
                this_vbgmm = self.vbgmm[t]
                for k_mix in range(n_mix):
                    cur_ch_like += this_vbgmm.weights_[k_mix] * norm.pdf(ground_truth[t:t + 1, i_ch:i_ch + 1],
                                                                         loc=this_vbgmm.means_[k_mix, i_ch],
                                                                         scale=np.sqrt(this_vbgmm.covariances_[k_mix,
                                                                                                               i_ch,
                                                                                                               i_ch]))
                ch_like += [cur_ch_like]
            all_ch_like += [np.concatenate(ch_like, axis=0)]

        return -np.log(np.concatenate(all_ch_like, axis=-1))

    def eval_marginal_like(self, x_ranges):
        all_ch_like = []
        n_mix = self.vbgmm[0].weights_.shape[0]

        for i_ch in range(self.n_channels):
            ch_like = []
            for t in range(self.draws.shape[1]):
                cur_ch_like = 0.
                this_vbgmm = self.vbgmm[t]
                for k_mix in range(n_mix):
                    cur_ch_like += this_vbgmm.weights_[k_mix] * norm.pdf(x_ranges[:, i_ch:i_ch+1],
                                                                         loc=this_vbgmm.means_[k_mix, i_ch],
                                                                         scale=np.sqrt(this_vbgmm.covariances_[k_mix,
                                                                                                               i_ch,
                                                                                                               i_ch]))
                ch_like += [cur_ch_like]
            all_ch_like += [np.stack(ch_like, axis=0)]

        return np.concatenate(all_ch_like, axis=-1)

    def eval_marginal_cdf(self, x_ranges):
        all_ch_like = []
        n_mix = self.vbgmm[0].weights_.shape[0]

        for i_ch in range(self.n_channels):
            ch_like = []
            for t in range(self.draws.shape[1]):
                cur_ch_like = 0.
                this_vbgmm = self.vbgmm[t]
                for k_mix in range(n_mix):
                    cur_ch_like += this_vbgmm.weights_[k_mix] * norm.cdf(x_ranges[:, i_ch:i_ch+1],
                                                                         loc=this_vbgmm.means_[k_mix, i_ch],
                                                                         scale=np.sqrt(this_vbgmm.covariances_[k_mix,
                                                                                                               i_ch,
                                                                                                               i_ch]))
                ch_like += [cur_ch_like]
            all_ch_like += [np.stack(ch_like, axis=0)]

        return np.concatenate(all_ch_like, axis=-1)


class TestDefinition(object):
    """Defines a ground truth and a metric to evaluate predictions on.
    Sequences of tests can be provided and reused across different model strategies, which guarantees result consistency.

    Args:
        metric_key (str): Name of the metric method to invoke on the provided predictions
        ground_truth (np.ndarray): True time series to compare forecasts with under the provided metric
        compare (function): Function to compare metrics (e.g. ascending or descending)
    """

    def __init__(self, metric_key, ground_truth, compare=None, eval_dict={}, id=''):
        self.metric = metric_key
        self.ground_truth = ground_truth
        self.eval_kwargs = eval_dict
        self.id = id
        if compare is None:
            self.compare = lambda x,y: x<y
        else:
            self.compare = compare

    def eval(self, prediction):
        """Evaluates the forecast """
        # type: (Prediction) -> float
        metric_eval = getattr(prediction, self.metric, None)

        if metric_eval is None or not callable(metric_eval):
            print('Metric {} is unavailable for this')
            result =  None
        else:
            result = metric_eval(self.ground_truth)

        return result
