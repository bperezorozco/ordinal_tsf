# MOrdReD: Ordinal autoregression with recurrent neural networks
## Introduction
This Python library accompanies our [work](https://arxiv.org/abs/1803.09704). MOrdReD enables time series forecasting in an autoregressive and ordinal fashion. This simply means that each new sample is forecasted by looking at the last previous observations for some lookback T.

Our framework provides an implementation of our ordinal autoregression framework (via Keras) described in the paper above; however, it also provides a flexible and amicable interface to set up time series forecasting tasks (parameter optimisation, model selection, model evaluation, long-term prediction, plotting) with either our prediction framework, or other well-established techniques, such as Gaussian Processes (via GPy) or Dynamic AR models (via statsmodels).
