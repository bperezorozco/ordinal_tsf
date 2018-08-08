from abc import ABCMeta, abstractmethod
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib.colors import ListedColormap
from util import assert_sum_one, all_satisfy, frame_ts, is_univariate, frame_generator, frame_generator_list
import pickle
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.spatial.distance import euclidean
from functools import partial
from sklearn.mixture import BayesianGaussianMixture
from fastdtw import fastdtw
import copy


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
    def __init__(self, ts_list, frame_length, p_train=0.7, p_val=0.15, p_test=0.15, preprocessing_steps=[]):
        # type: (List[np.ndarray], int, float, float, float, List[DatasetPreprocessingStep]) -> TimeSeriesSetDataset

        assert [raw_ts.ndim == 2 and raw_ts.shape[0] > frame_length for raw_ts in ts_list], \
            'Please provide only univariate time series as input to Dataset.'

        self.raw_ts_list = []
        self.frame_length = frame_length
        self.train_ts = []
        self.val_ts = []
        self.test_ts = []
        self.optional_params_list = []
        self.preprocessing_steps_list = []
        self.frame_length = frame_length

        for ts in ts_list:
            self.raw_ts_list += [ts.copy()]
            n_train = int(p_train * ts.shape[0])
            n_train_val = int((p_train + p_val) * ts.shape[0])

            cur_train_ts = ts[:n_train]
            cur_val_ts = ts[n_train:n_train_val]
            cur_test_ts = ts[n_train_val:]
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

        self.train_frames = partial(frame_generator_list, ts_list=self.train_ts, frame_length=frame_length)
        self.val_frames = partial(frame_generator_list, ts_list=self.val_ts, frame_length=frame_length)
        self.test_frames = partial(frame_generator_list, ts_list=self.test_ts, frame_length=frame_length)


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
        self.mean = ts.mean()
        self.std = ts.std()
        self.param_dict = {'ts_mean': self.mean,
                           'ts_std': self.std,
                           'zero_mean_unit_var':True}
        self.is_fitted = True


class Quantiser(DatasetPreprocessingStep):
    """Computes ordinal bins and allocates each observation in the time series."""
    def __init__(self, n_bins=None, delta=1e-3):
        self.n_bins = n_bins
        self.delta = delta
        self.bins = None

    def apply(self, ts):
        if not self.is_fitted:
            self.fit(ts)
        assert is_univariate(ts), 'Only univariate time series can be quantised. Current shape: {}'.format(ts.shape)

        out = np.zeros((ts.shape[0], self.n_bins))
        digits = np.searchsorted(self.bins[:-1], ts.squeeze())

        for i, i_d in enumerate(digits):
            out[i, i_d] = 1.

        return out

    def fit(self, ts):
        ts_max = ts.max()
        ts_min = ts.min()

        if self.n_bins is None:
            self.n_bins = self.__find_n_bins(ts)

        self.bins = np.linspace(ts_min, ts_max, self.n_bins)

        self.param_dict = {'bins': self.bins,
                           'bin_delta':self.bins[1]-self.bins[0],
                           'is_ordinal': True,
                           'frame_generator': True}

        self.is_fitted = True

    def __find_n_bins(self, ts):
        # type: (np.ndarray) -> int
        MAX_ALLOWED = 300
        MIN_ALLOWED = 50
        n_bins = np.unique(ts.squeeze()).shape[0]

        if n_bins < MAX_ALLOWED and n_bins > MIN_ALLOWED:
            return n_bins

        ts_max = ts.max()
        ts_min = ts.min()
        n_bins = int((ts_max - ts_min) / self.delta)
        n_bins = max(min(MAX_ALLOWED, n_bins), MIN_ALLOWED)

        return n_bins


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
        plt.legend(['Quantile 0.025', 'Quantile 0.975', 'Median', 'True'])

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
        #plt.title('Empirical distribution function')
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


class TestDefinition(object):
    """Defines a ground truth and a metric to evaluate predictions on.
    Sequences of tests can be provided and reused across different model strategies, which guarantees result consistency.

    Args:
        metric_key (str): Name of the metric method to invoke on the provided predictions
        ground_truth (np.ndarray): True time series to compare forecasts with under the provided metric
        compare (function): Function to compare metrics (e.g. ascending or descending)
    """

    def __init__(self, metric_key, ground_truth, compare=None):
        self.metric = metric_key
        self.ground_truth = ground_truth
        if compare is None:
            self.compare = lambda x,y: x<y
        else:
            self.compare = compare

    def eval(self, prediction):
        """Evaluates the forecast """
        # type: (Prediction) -> float
        metric_eval = getattr(prediction, self.metric, None)

        if metric_eval is None or not callable(metric_eval):
            print 'Metric {} is unavailable for this'
            result =  None
        else:
            result = metric_eval(self.ground_truth)

        return result
