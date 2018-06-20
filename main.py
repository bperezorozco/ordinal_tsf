import os
import pandas as pd
from ordinal_tsf.model import MordredStrategy
from ordinal_tsf.dataset import Dataset, Quantiser, Standardiser, WhiteCorrupter, AttractorStacker, Selector, TestDefinition
from ordinal_tsf.session import Session
from ordinal_tsf.util import cartesian
os.environ["CUDA_VISIBLE_DEVICES"]="4,5"


DS = 'mg'
MAX_LENGTH = 30000
TS_X = 100
TS_Y = 100
LOOKBACK = 100
HORIZON = 100
ATTRACTOR_LAG = 10
VALIDATION_START_INDEX = LOOKBACK + 2*ATTRACTOR_LAG + 1
VALIDATION_PREDICTIVE_HORIZON = 500
TEST_START_INDEX = LOOKBACK + 2*ATTRACTOR_LAG + 1
TEST_PREDICTIVE_HORIZON = 1000
# open/create session (folder name) (includes raw time series loading)
# choose experiment (includes model preparation)
best_models = {}
mordred_search_space = {'lambda': [1e-6, 1e-7, 1e-8],
                        'dropout_rate': [0.25, 0.5],
                        'units': [64, 128, 256, 320],
                        'lookback': [LOOKBACK],
                        'horizon': [HORIZON]}

train_spec = {'epochs': 50, 'batch_size': 256, 'validation_split': 0.15}

for DS in ['heart', 'tide', 'mg']:
    sess = Session('{}'.format(DS))
    x = pd.read_feather('../ds/{}.feather'.format(DS)).values[:MAX_LENGTH]
    stand = Standardiser()
    quant = Quantiser()
    selector = Selector(VALIDATION_START_INDEX, VALIDATION_PREDICTIVE_HORIZON)
    white_noise = WhiteCorrupter()
    att_stacker = AttractorStacker(10)

    dataset = Dataset(x, LOOKBACK + HORIZON + 1, p_val=0.15, p_test=0.15,
                      signal_preprocessing_steps=[stand],
                      structural_preprocessing_steps=[quant])
    continuous_ground_truth = dataset.apply_partial_preprocessing('val', [selector, stand])
    ordinal_ground_truth = dataset.apply_partial_preprocessing('val', [selector, stand, quant])
    validation_tests = [TestDefinition('mse', continuous_ground_truth),
                        TestDefinition('nll', ordinal_ground_truth)]
    validation_plots = {'plot_median_2std': {'ground_truth':continuous_ground_truth},
                        'plot_like': {}}

    mordred_search_space['ordinal_bins'] = [quant.n_bins]

    eval_spec = {'mc_samples': 100}

    experiment = sess.start_experiment(dataset, MordredStrategy)
    best_models[DS] = experiment.choose_model(validation_tests,
                                              mordred_search_space.keys(),
                                              cartesian(mordred_search_space.values()),
                                              VALIDATION_START_INDEX,
                                              VALIDATION_PREDICTIVE_HORIZON,
                                              plots=validation_plots,
                                              fit_kwargs=train_spec,
                                              eval_kwargs=eval_spec,
                                              mode='val')

    print "Done with: {}".format(DS)
    print best_models

    selector = Selector(TEST_START_INDEX, TEST_PREDICTIVE_HORIZON)
    continuous_ground_truth = dataset.apply_partial_preprocessing('test', [selector, stand])
    ordinal_ground_truth = dataset.apply_partial_preprocessing('test', [selector, stand, quant])
    final_tests = [TestDefinition('mse', continuous_ground_truth),
                   TestDefinition('nll', ordinal_ground_truth)]
    test_plots = {'plot_median_2std': {'ground_truth': continuous_ground_truth},
                  'plot_like': {}}

    for metric, best_model in best_models[DS].items():
        print 'Test result for model with best {} performance'.format(metric)
        experiment.choose_model(final_tests,
                                best_model.keys(),
                                [best_model.values()],
                                TEST_START_INDEX,
                                TEST_PREDICTIVE_HORIZON,
                                plots=test_plots,
                                fit_kwargs=train_spec,
                                eval_kwargs=eval_spec,
                                mode='test')


exit()

