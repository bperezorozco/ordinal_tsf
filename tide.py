import os
import pandas as pd
from ordinal_tsf.model import MordredStrategy
os.environ["CUDA_VISIBLE_DEVICES"]="3"
from ordinal_tsf.dataset import Dataset, Quantiser, Standardiser, WhiteCorrupter, AttractorStacker, Selector, TestDefinition
from ordinal_tsf.session import Session

DS = 'tide'
MAX_LENGTH = 30000
TS_X = 100
TS_Y = 100
LOOKBACK = 100
HORIZON = 100
ATTRACTOR_LAG = 10
VALIDATION_START_INDEX = LOOKBACK + 2*ATTRACTOR_LAG + 1
VALIDATION_PREDICTIVE_HORIZON = 500
# open/create session (folder name) (includes raw time series loading)
# choose experiment (includes model preparation)
best_models = {}
mordred_search_space = {'lambda': [1e-7, 1e-8, 1e-9],
                        'dropout_rate': [0.25, 0.5],
                        'units': [64, 128, 256],
                        'lookback': [LOOKBACK],
                        'horizon': [HORIZON]}

train_spec = {'epochs': 50, 'batch_size': 256, 'validation_split': 0.15}

sess = Session('{}'.format(DS))
x = pd.read_feather('../ds/{}.feather'.format(DS)).values[:MAX_LENGTH]
stand = Standardiser()
quant = Quantiser()
selector = Selector(VALIDATION_START_INDEX, VALIDATION_PREDICTIVE_HORIZON)
white_noise = WhiteCorrupter()
att_stacker = AttractorStacker(10)

mordred_search_space['ordinal_bins'] = [quant.n_bins]

dataset = Dataset(x, LOOKBACK + HORIZON + 1, p_val=0.15, p_test=0.15,
                  signal_preprocessing_steps=[stand],
                  structural_preprocessing_steps=[quant])
continuous_ground_truth = dataset.apply_partial_preprocessing('val', [selector, stand])
ordinal_ground_truth = dataset.apply_partial_preprocessing('val', [selector, stand, quant])
validation_tests = [TestDefinition('mse', continuous_ground_truth),
                    TestDefinition('nll', ordinal_ground_truth)]
validation_plots = {'plot_median_2std': {'ground_truth':continuous_ground_truth},
                    'plot_like': {}}

eval_spec = {'mc_samples': 100}

experiment = sess.start_experiment(dataset, MordredStrategy)
best_models[DS] = experiment.choose_model(validation_tests,
                                          mordred_search_space,
                                          VALIDATION_START_INDEX,
                                          VALIDATION_PREDICTIVE_HORIZON,
                                          plots=validation_plots,
                                          fit_kwargs=train_spec,
                                          eval_kwargs=eval_spec)

print "Done with: {}".format(DS)


exit()

