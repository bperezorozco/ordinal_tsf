from abc import ABCMeta, abstractmethod
from sklearn.metrics import mean_squared_error
from matplotlib.colors import ListedColormap
from util import assert_sum_one, all_satisfy, frame_ts, is_univariate
import pickle
import numpy as np
import seaborn as sns
from scipy.stats import norm


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
            self.n_bins = self.__find_n_bins(ts_max, ts_min)

        self.bins = np.linspace(ts_min, ts_max, self.n_bins)

        self.param_dict = {'bins': self.bins,
                           'bin_delta':self.bins[1]-self.bins[0],
                           'is_ordinal': True}

        self.is_fitted = True

    def __find_n_bins(self, ts_max, ts_min):
        n_bins = int((ts_max - ts_min) / self.delta)
        n_bins = max(min(300, n_bins), 80)
        print n_bins
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

    def mse_and_std(self, ground_truth):
        """Computes MSE +- STD between two real-valued time series"""
        # type: (np.ndarray) -> np.float
        all_mse = [mean_squared_error(ground_truth, prediction) for prediction in self.draws]
        return np.mean(all_mse), np.std(all_mse)

    def nll(self, binned_ground_truth):
        """Computes NLL of drawing a time series from a piecewise uniform sequential prediction"""
        # type: (np.ndarray) -> np.float
        p_ground_truth = (self.ordinal_pdf * binned_ground_truth / self.delta).max(axis=-1)
        neg_log_p_ground_truth = -np.log(p_ground_truth)
        return neg_log_p_ground_truth.sum()

    def cum_nll(self, binned_ground_truth):
        """Computes integral of NLL(t) of drawing a time series from a piecewise uniform sequential prediction"""
        # type: (np.ndarray) -> np.float
        p_ground_truth = (self.ordinal_pdf * binned_ground_truth / self.delta).max(axis=-1)
        neg_log_p_ground_truth = -np.log(p_ground_truth)
        return neg_log_p_ground_truth.cumsum().sum()

    def get_quantile(self, alpha):
        """Computes \alpha-quantiles given the object's ordinal pdf"""
        # type: (np.ndarray) -> np.float
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

    def plot_like(self, plt):
        """Plots the full ordinal pdf as a heatmap"""
        plt.imshow(self.ordinal_pdf.T, cmap=ListedColormap(sns.color_palette("RdBu_r", 500).as_hex()),
                   origin='lower', aspect='auto')

    def plot_cum_nll(self, plt, binned_ground_truth):
        """Plots the full ordinal pdf as a heatmap"""
        p_ground_truth = (self.ordinal_pdf * binned_ground_truth / self.delta).max(axis=-1)
        neg_log_p_ground_truth = -np.log(p_ground_truth)
        cum_nll = neg_log_p_ground_truth.cumsum()

        plt.plot(cum_nll)
        plt.title('Cumulative negative log likelihood')

    def plot_log_like(self, plt):
        """Plots the full log ordinal pdf as a heatmap"""
        log_like = np.ma.log(self.ordinal_pdf.T)
        plt.imshow(log_like, cmap=ListedColormap(sns.color_palette("RdBu_r", 500).as_hex()),
                   origin='lower', aspect='auto')


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

    def __init__(self, draws):
        self.posterior_mean = draws.mean(axis=0)
        self.posterior_std = draws.std(axis=0)
        self.draws = draws

    def mse(self, ground_truth):
        """Computes MSE between two real-valued time series"""
        # type: (np.ndarray) -> np.float
        return np.mean([mean_squared_error(ground_truth, p) for p in self.draws])

    def nll(self, ground_truth):
        """Computes NLL of drawing a time series from a GP sequential prediction"""
        # type: (np.ndarray) -> np.float
        horizon = self.posterior_mean.shape[0]
        likelihood = np.array([norm(loc=self.posterior_mean[i], scale=self.posterior_std[i]).pdf(ground_truth[i])
                               for i in range(horizon)])

        nll = -np.log(likelihood).sum()
        print 'NLL: {}'.format(nll)
        return nll

    def cum_nll(self, ground_truth):
        """Computes compulative NLL of drawing a time series from a GP sequential prediction"""
        # type: (np.ndarray) -> np.float
        horizon = self.posterior_mean.shape[0]
        likelihood = np.array([norm(loc=self.posterior_mean[i], scale=self.posterior_std[i]).pdf(ground_truth[i])
                               for i in range(horizon)])

        nll = -np.log(likelihood).cumsum().sum()
        print 'Cum NLL: {}'.format(nll)
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


class TestDefinition(object):
    """Defines a ground truth and a metric to evaluate predictions on.

    By defining a sequence of tests, we

    Args:
        metric_key (str): Name of the metric method to invoke on the provided predictions
        ground_truth (np.ndarray): True time series to compare forecasts with under the provided metric
        compare (function):
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
