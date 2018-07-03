import pickle
import os
from dataset import Dataset, OrdinalPrediction, GaussianPrediction, TestDefinition, Prediction
from model import ModelStrategy
import numpy as np
from dataset import Prediction
import matplotlib.pyplot as plt


class Session:
    """This is the director of all experiments made with a single dataset.

    All raw time series are associated to a single session, which manages the logs, predictions, plots and reports
    of different forecasting strategies and dataset representations (real-valued, ordinal, stacked, etc.)"""
    def __init__(self, directory):
        if directory[-1] != '/':
            directory += '/'

        self.directory = directory
        self.experiments = []
        self.dataset = []

        if not os.path.isdir(directory):
            print 'Creating new session: {}...'.format(directory)
            os.makedirs(directory)
            os.makedirs(directory + 'datasets/')


        print 'Opening session: {}...'.format(directory)
        self.experiments_directory = directory + 'experiments/'
        self.datasets_directory = directory + 'datasets/'

    def start_experiment(self, dataset, StrategyClass):
        """This initiates a new experiment with a forecasting strategy"""
        if not os.path.isdir(self.directory + StrategyClass.id):
            os.makedirs(self.directory + StrategyClass.id)
            os.makedirs(self.directory + StrategyClass.id + '/models/')
            os.makedirs(self.directory + StrategyClass.id + '/logs/')
            os.makedirs(self.directory + StrategyClass.id + '/predictions/')
            os.makedirs(self.directory + StrategyClass.id + '/plots/')

        dataset.save(self.directory + 'datasets/' + dataset.get_default_fname(self.directory[:-1]))
        #dataset = Dataset.load(self.directory + 'datasets/' + dataset.get_default_fname(self.directory[:-1]))
        return StrategyExperiment(dataset, self.directory + StrategyClass.id + '/', StrategyClass)


class StrategyExperiment:
    """This manages a strategy's fitting, hyperparameter selection and evaluation.
    This acts as a directory handler that queries, stores and manages the different products of an experiment:
    parameter fitting, hyperparameter selection, model evaluation and plot generation."""
    def __init__(self, dataset, folder, StrategyClass):
        self.dataset = dataset  # type: Dataset
        self.all_specs = {}  # type: dict
        self.folder = folder  # type: str
        self.Strategy = StrategyClass  # type: class

    def build_prediction(self, prediction):
        """Adapts the raw prediction of a strategy into the correct format as given by the dataset's representation"""
        # type: (dict) -> Prediction
        if self.dataset.optional_params.get('is_attractor', False):
            for k, v in prediction.items():
                prediction[k] = v.take(-1, axis=-1)

        if self.dataset.optional_params.get('is_ordinal', False):
            prediction['bins'] = self.dataset.optional_params['bins']
            return OrdinalPrediction(**prediction)

        return GaussianPrediction(**prediction)

    def choose_model(self, tests, hypergrid_keys, expanded_hypergrid, prediction_index,
                     predictive_horizon, plots=[], fit_kwargs={}, eval_kwargs={}, mode='val'):
        """Executes the model selection and evaluation pipeline
        Args:
            tests (List[TestDefinition]): criteria and targets used to evaluate predictions
            hypergrid_keys (List[str]): names of the Strategy's attributes
            expanded_hypergrid (list): hyperparameter configurations to be evaluated
            prediction_index (int): time index of the first prediction to be obtained
            predictive_horizon (int): length of the forecast
            plots (List[str]): plots to be requested from obtained predictions
            fit_kwargs (dict): optional parameters to be passed on the Strategy's fit method
            eval_kwards (dict): optional parameters to be passed on the Strategy's predict method
            mode (str): whether the method should use the validation or test data
        """
        # type: (List[TestDefinition], dict, int, int, dict, dict, dict) -> {}
        best_results = {}
        best_strategy = {}
        eval_ts = self.dataset.val_ts if mode == 'val' else self.dataset.test_ts

        for test in tests:
            best_results[test.metric] = None
            best_strategy[test.metric] = None

        for next_config in expanded_hypergrid:
            spec = {k:v for k, v in zip(hypergrid_keys, next_config)}
            fname = self.Strategy.get_filename(spec)
            model_fname = self.folder + 'models/' + fname
            spec_fname = self.folder + 'logs/' + fname + '_{}_report'.format(mode)
            prediction_fname = self.folder + 'predictions/' + fname \
                             + '_{}_pred_index_{}_pred_horizon_{}'.format(mode, prediction_index, predictive_horizon)

            if os.path.isfile(spec_fname):
                with open(spec_fname, 'r') as f:
                    spec = pickle.load(f)

            if os.path.isfile(prediction_fname):
                with open(prediction_fname, 'rb') as f:
                    prediction = pickle.load(f)
            else:
                if os.path.isfile(model_fname):
                    model = self.Strategy.load(model_fname)
                else:
                    print 'Training new model with specification: {}'.format(spec)
                    model = self.Strategy(**spec)
                    model.fit(self.dataset.train_frames, **fit_kwargs)
                    model.save(self.folder + 'models/')

                seed_start = prediction_index - model.seed_length
                assert seed_start >= 0, \
                    "Prediction index {} must be greater than model strategy seed length".format(prediction_index,
                                                                                                 model.seed_length)
                model_input = eval_ts[np.newaxis, seed_start:prediction_index]

                prediction = model.predict(model_input,
                                           predictive_horizon=predictive_horizon,
                                           **eval_kwargs)
                prediction = self.build_prediction(prediction)

                with open(prediction_fname, 'wb') as f:
                    pickle.dump(prediction, f)

            for test in tests:
                metric = test.metric
                if metric not in spec:
                    spec[metric] = test.eval(prediction)

                if test.compare(spec[metric], best_results[metric]) or best_results[metric] is None:
                    print 'NEW BEST MODEL'
                    print metric, spec[metric]
                    best_strategy[metric] = spec
                    best_results[metric] = spec[metric]

            with open(spec_fname, 'wb') as f:
                pickle.dump(spec, f)

            for plot_key, plot_args in plots.items():
                plot_eval = getattr(prediction, plot_key, None)

                if plot_eval is None or not callable(plot_eval):
                    print "Plot {} unavailable for this prediction.".format(plot_key)
                    continue

                plot_eval(plt, **plot_args)
                plot_fname = '{}{}_{}_{}.pdf'.format(self.folder + 'plots/', plot_key, mode, fname)
                plt.savefig(plot_fname, format='pdf')
                plt.clf()

        print 'Best performance and strategies: '
        for test in tests:
            print test.metric, best_results[test.metric], best_strategy[test.metric]

        return best_strategy
