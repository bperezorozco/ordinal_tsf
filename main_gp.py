import os
import pandas as pd
from ordinal_tsf.model import GPStrategy
from ordinal_tsf.dataset import Dataset, Standardiser, WhiteCorrupter, AttractorStacker, Selector, TestDefinition
from ordinal_tsf.session import Session
from ordinal_tsf.util import cartesian
import GPy as gpy
os.environ["CUDA_VISIBLE_DEVICES"]="5"


MAX_LENGTH = 30000
TS_X = 100
TS_Y = 100
LOOKBACK = 100
HORIZON = 100
ATTRACTOR_LAG = 10
EFFECTIVE_LAG = 0
VALIDATION_PREDICTIVE_HORIZON = 500
TEST_PREDICTIVE_HORIZON = 1000
# open/create session (folder name) (includes raw time series loading)
# choose experiment (includes model preparation)
best_models = {}

gp_search_space = {'ker':[gpy.kern.Matern52(LOOKBACK, ARD=True)], 'fname':['matern_32_white']}
train_spec = {}

VALIDATION_START_INDEX = LOOKBACK + 2 * ATTRACTOR_LAG + 1
TEST_START_INDEX = LOOKBACK + 2 * ATTRACTOR_LAG + 1

for DS in ['webtsslp', 'EMexptqp2']:
    sess = Session('{}'.format(DS))
    x = pd.read_feather('../ds/{}.feather'.format(DS)).values[:MAX_LENGTH]
    stand = Standardiser()
    white_noise = WhiteCorrupter(1e-2)
    att_stacker = AttractorStacker(10)

    dataset = Dataset(x, LOOKBACK + HORIZON + 1, p_val=0.15, p_test=0.15, preprocessing_steps=[stand, white_noise])
    if dataset.optional_params.get('is_attractor', False):
        EFFECTIVE_LAG = 2*ATTRACTOR_LAG
    gp_search_space['n_channels'] = [dataset.optional_params.get('n_channels', 1)]

    selector = Selector(VALIDATION_START_INDEX, VALIDATION_PREDICTIVE_HORIZON)
    continuous_ground_truth = dataset.apply_partial_preprocessing('val', [selector, stand])

    validation_tests = [TestDefinition('mse', continuous_ground_truth),
                        TestDefinition('nll', continuous_ground_truth)]
    validation_plots = {'plot_median_2std': {'ground_truth':continuous_ground_truth}}

    experiment = sess.start_experiment(dataset, GPStrategy)
    best_models[DS] = experiment.choose_model(validation_tests,
                                              gp_search_space.keys(),
                                              cartesian(gp_search_space.values()),
                                              VALIDATION_START_INDEX - EFFECTIVE_LAG,
                                              VALIDATION_PREDICTIVE_HORIZON,
                                              plots=validation_plots,
                                              fit_kwargs=train_spec,
                                              eval_kwargs={'mc_samples': 100},
                                              mode='val')

    print "Done with: {}".format(DS)
    print best_models

    selector = Selector(TEST_START_INDEX, TEST_PREDICTIVE_HORIZON)
    continuous_ground_truth = dataset.apply_partial_preprocessing('test', [selector, stand])
    ordinal_ground_truth = dataset.apply_partial_preprocessing('test', [selector, stand])

    final_tests = [TestDefinition('mse', continuous_ground_truth),
                   TestDefinition('nll', continuous_ground_truth)]
    test_plots = {'plot_median_2std': {'ground_truth': continuous_ground_truth}}

    for metric, best_model in best_models[DS].items():
        print 'Test result for model with best {} performance'.format(metric)
        experiment.choose_model(final_tests,
                                best_model.keys(),
                                [best_model.values()],
                                TEST_START_INDEX - EFFECTIVE_LAG,
                                TEST_PREDICTIVE_HORIZON,
                                plots=test_plots,
                                fit_kwargs=train_spec,
                                eval_kwargs={'mc_samples': 100},
                                mode='test')

print best_models

exit()

