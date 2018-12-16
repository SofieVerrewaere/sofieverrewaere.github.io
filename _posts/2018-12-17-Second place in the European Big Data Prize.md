---
layout: post
title: "European Commission Horizon 2020 prize on Big Data Technologies"
subtitle:   "Second place and €400.000 award!"
date:       2018-12-17 00:00:00
author:     "Sofie Verrewaere"
header-img: "img/EC/front.jpg"
comments: true
---

## Overview
<a href="https://ec.europa.eu/digital-single-market/en/news/big-data-prizes-awarded-most-accurate-electricity-grid-flow-predictions" target="_blank"><b>I finished joint second in the Big Data Technologies H2020 Prize!</b></a>

The goal of this blog post is to explain my approach to a very challenging problem. I will cover the high level approach in design decissions that enabled me to be succesful. 
Here's an overview of the different sections. If you want to skip ahead, just click the section title to go there.

* *[Introduction](#introduction)*
* *[Model Approach](#modelApproach)*
   * *[Pre-Processing](#preProcessing)*
   * *[Feature engineering](#featEng)*
   * *[Model Architecture](#modelArchitecture)*
* *[Closing Remarks](#closingRemarks)*

## <a name="introduction"><a> Introduction

First of all I would like to thank the European Commission (EC) for organizing the Big Data Technologies H2020 Prize. It has been an amazing experience! I would also like to thank the other participants and congratulate the other winners, José Vilar (1ste Prize Winner) and Yann-Aël Le Borgne for their remarkable achievement!

### Why this prize?
The context of the Big Data Technologies Horizon Prize can be found on the <a href="http://ec.europa.eu/research/horizonprize/index.cfm?prize=bigdata" target="_blank"><b>competition page:</b></a>

Many issues impacting society such as climate change, overcrowded and polluting transportation systems, wasteful energy consumption, would improve with our ability to examine historical records and predict the evolution of these different developments in our society and our economy.

Access to information in a timely way could have a positive impact on the way we consume energy, the way we organize our transport systems and even the way we run our health and other public services. Predicting consumption patterns must be accurate, and need to be delivered in real-time which will allow for effective action such as the use of energy resources in an optimal fashion. Faster, more accurate and resource-efficient prediction methods will result in more efficient management of sectors of the economy. Indeed today, these predictions are not always accurate and lead to the wasteful consumption of energy resources.

The European Commission has launched the Big Data Technologies Horizon Prize to address this issue. The goal is to develop a new spatiotemporal forecasting method which are able to beat ones currently available.

<a href="http://ec.europa.eu/research/participants/data/ref/h2020/other/prizes/contest_rules/h2020-prizes-rules-big-data_en.pdf" target="_blank"><b>Read the detailed rules of the contest here. </b></a>

### Challenge?
The challenge is to improve the performance of software for the forecasting of geospatio-temporal data (collections of time-stamped records that are linked to a geospatial location).  The prize rewards a solution which improves existing methods in terms of scalability, accuracy, speed and use of computational resources.

The solutions are ranked based on the accuracy of the predictions, expressed as <b>"root-mean square error" (RMSE) </b> and the speed of delivery of the predictions expressed as the <b>"overall elapsed execution time" (OEET)</b>. The participants are restricted in resources by a <a href="https://aws.amazon.com/ec2/instance-types/p2/" target="_blank">single "P2.8xlarge" AWS instance</a>.

The actual challenge requires the submission of code that can handle panel data. Panel data consists of multi-dimensional data involving measurements over time. The time series are split up in time in a <b>train</b> and an <b>adapt</b> phase. The adapt phase simulates new incoming data by progressively making the adapt data available. In total there are 600 adapt files that are contiguous in time. These data sets contain electrical flow time series with an interval of 5 minutes. In this challenge the aim is to forecast the flow of the next 60 min (1 hour), resulting in a forecast horizon of 12 steps. 

Auxiliary data sets were provided, but I decided not to use these in the final submission as no strong guarantees were given that the auxiliary data would be available at the prediction time. No information on the spatial component was given, therefore no spatial component could be taken into account.

The EC provided the contest platform on which the working software submissions are run against the <b>test data</b>. The testing data used at the contest platform was not accessible to the participants and only three submissions were allowed to verify the correct working of the code. This was actually the hardest challenge since particular care needed to be taken to avoid bugs on the contest platform. The contest platform measures the performance (accuracy, speed, with limited resources to the AWS instance) of each working software submission and was used for testing, and to score and pre-rank the participants' working software. Yet another data set, verification data (data from the same process, but at a different time period) was used for the verification runs and final ranking of the pre-selected applications by the jury.

Prior to the opening of the contest platform, a starting kit was provided. This is a simulator of the contest platform that allows a participant to get familiar with the contest running environment and allows for the testing of the working software against sample datasets, representative to the actual test
dataset. Specifically, the starting kit consists of 1916 time series with a train period of just over one year and an adapt period of approximately three months.

One of the difficulties of this challenge is the unknown test distribution, which makes it very difficult to improve the model. As little is known of the unseen test data conservative parameter settings are used in the final submission. Lot's of inductive biases are used to ease the heavy lifting of the required model. These inductive biases are stressed throughout the Model Approach.

## <a name="modelApproach"><a> Model Approach

This section aims to summarize the modeling strategy of the competition approach on a high level. All logic is implemented in <b>Python 2</b>, which is GPL compatible. <b>Tensorflow</b> is used handling deep learning in Python. The Python package <b>Pandas</b> is used for tabular data handling and the Python package <b>Numpy</b> deals with for numeric computations. The Python package <b>Multiprocessing</b> is used to parallelize the computations on CPU.

### <a name="preProcessing"><a> Pre-processing

The most important steps of the data preprocessing approach are discussed below. 
The motivation to perform the pre-processing as described below is to allow disentangling the major factors of variation observed in the starting kit data (long zero periods and true variation) with a single neural net.
The raw data contains all sorts of unusual time series patterns of which the following three are the most important: <b>Short outlier bursts</b>, <b>Interpolated values</b> and <b>Zero values<b/>. All pre-processing steps use relative magnitude hyperparameters resulting in a scale invariant procedure.


#### Outliers
Short burst outliers are removed from the training data and are ignored completely since it is likely to hurt the modeling capability. The better fit is expected to outweigh benefits from learning about outlier patterns.
{% include image.html url="/img/EC/remove_outliers.jpg" description="<small>Removing Outliers</small>" %}


#### Interpolated values
The interpolated values are an artefact of the preprocessing logic in the starting kit. Therefore I decided to NOT include interpolated data points. Including interpolated values would encourage the model to learn to continue the interpolation to the next real data point but this next real data point will obviously not be available when considering future data!
{% include image.html url="/img/EC/remove_interpolations.jpg" description="<small>Removing Interpolations</small>" %}


#### Zero values
Zero values occur frequently in the training data (about one in three data points). Unlike the interpolated values, these represent real target values and hence require a special treatment. The huge amount of zero values is captured by defining two types of targets, targets for regular (non-zero) values and the probability of zero values. The combined forecast is the probability of a non-zero value times the forecasted non-zero value, conditioned on it not being zero. This corresponds to the combined regression target in expectation.
Modeling the two losses independently is thus tackling the same objective of optimizing the RMSE with a single model! 
When the target is zero, the regular values target is unknown and is not incorporated in the optimization of the regular values loss.
{% include image.html url="/img/EC/split_into_2_series.jpg" description="<small>Split time series into two series, one for zero values (0/1), one for regular values[0,1]</small>" %}


#### Missing values
It was unclear if actual missing values could be present in the raw data. Should they occure, missing values are interpolated before the pre-processing. This results in those data points being ignored in the model fitting.


#### Scale, transform and augment
Neural networks work better when the inputs are normalized. Techniques like batch-norm handle inputs/internal network covariates that have constant variance but it is likely to be better if the inputs are normalized. The input series were normalized to [0, 1] after exclusion of the outliers. I also augmented this normalized data by containing lags of the input, changes in input time steps and by adding a binary mask to indicate if the input data point is a missing value.

The advantage of rescaling the time series is that one shared model can be used to predict all the time series. 

The way the loss is formulated can heavily impact the performance of the model. Benchmarks on the starting kit data showed that predicting the change versus the last non missing value works significantly better than predicting next values. This simple change biases the model to predict no change and reserves modeling capacity to focus on true factors of variation.

Incorporating features of the time of the day also showed to help benchmarks significantly. These include periodical features on the day and the week. I could have used hour and year too, but dropped those out of fear for overfitting. Features are calculated by a polar transformation.
Series that are zero or missing for most of the individual series will be treated differently, this is further discussed in the input of the deep learning model.

#### Input Deep Learning Model

In this section the inputs of the neural network are described. The term missing values refers to values that belong to the regular values that are neither interpolated or outliers.

Not all the time series are used to feed the deep learning model. The time series are subdivided in valid and invalid data, based on the number of missing values. Series consisting of more than 90% of missing values in the train phase are considered invalid. However, the status (valid/invalid) can change in the adapt phase. Three possible scenarios are considered for each series independently at each adapt step (displayed as 1, 2 and 3 in following figures). 
{% include image.html url="/img/EC/splitdata.jpg" description="<small>Split time series in valid and invalid data</small>" %}

* <b>Scenario 1</b>: Scenario 1 applies to series which are valid at both times (train and adapt phase). The number of missing values are limited and the regime (range in particular) is consistent throughout time. The time series are subjected to two types of manipulations, scaling and differentiating. The scaling parameters (max, min,...) are determined in the training phase and are stored between prediction steps. In the adapt phase the scaling parameters are reloaded and used to scale the adapt time series.
{% include image.html url="/img/EC/scenario1.jpg" description="<small>Scenario 1</small>" %}


* <b>Scenario 2</b>: Scenario 2 applies to series which are valid in the training phase, but act different in the adapt phase (e.g. the ranges (min - max) change). The scaling determined in the training phase is no longer valid. If the scaled series exceed 1.2 or drop below -0.2 the series are considered temporary invalid. During temporary invalidness predictions are persistence. Persistence predictions are extrapolations of the last non outlier or interpolated value. If the last value is a 0 the persistence prediction is a zero as well.
{% include image.html url="/img/EC/scenario2.jpg" description="<small>Scenario 2</small>" %}


* <b>Scenario 3</b>: Scenario 3 applies to series which are invalid in the training phase, but become active in the adapt phase. Scaling parameters are determined periodically in the adapt phase after 150 steps. 
{% include image.html url="/img/EC/scenario3.jpg" description="<small>Scenario 3</small>" %}

In the training phase, all possible data was taken into account. In the adapt phase, a <b>moving window approach </b>is applied. Only a part of the historical data is taken into account to perform the prediction. The number of steps and the forecast horizon are predetermined by the EC.
{% include image.html url="/img/EC/moving_window.jpg" description="<small>Moving window approach in the Adapt phase</small>" %}

The deep learning model can be (but is not in the final submission) (pre-)trained and updated on specific moments in time. The pre-training could have been done partially. The best candidates for the pre-training would be the first layers of the three considered MLP's. This first layers are more likely to detect generic time series features and are most likely to transfer well to new data.
All data is used in the training phase, with a limited train window, to train the model and the scaling parameters are determined (cfr. Scenario 1). In the adapt phase, the <b>model</b> (and scaling parameters in Scenario 3) <b>gets updated, every fixed number of steps</b> (150 steps).
{% include image.html url="/img/EC/model_update.jpg" description="<small>Data Flow</small>" %}

### <a name="featEng"><a> Feature engineering

To summarize, the considered predictors used in the final model are:

* The current value and a flag if it is missing value
* Periodical features of the part of the day (sin/cos projection)
* Periodical features of the part of the week (sin/cos projection)
* 33 Lagged values [1, 2, 3, 4, 5, 7, 10, 14, 20, 28, 36, 48, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 270, 280, 283, 285, 287, 288, 289, 291, 293, 296]: First and last lag are absolute scaled values[0, 1], others are relative changes of the scaled lagged values. There is a concentration of lag values at 1 day back as many series showed daily periodic trends (24 * 12 = 288 
* 5 minutes).
* 33 Lagged missing values [1, 2, 3, 4, 5, 7, 10, 14, 20, 28, 36, 48, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 270, 280, 283, 285, 287, 288, 289, 291, 293, 296]: True/False/missing values
* Last not missing value: relative change of the lagged values/missing value
* Number of consecutive zeros: scaled [0,1]:
   * 0: no zeros
   * Number of consecutive zeros/300 for 0 < Number of consecutive zeros < 300
   * 1 otherwise
  

### <a name="modelArchitecture"><a> Model Architecture

The deep learning model consists of three different MLP’s and one shared optimizer. These three different MLP's are combined into on single model that is used for all time series to leverage the generalization capabilities of the neural networks to handl the complex modeling problem.

{% include image.html url="/img/EC/model_architecture.jpg" description="<small> Model Architecture </small>" %}


* <b>Embedding MLP</b>: Aims to extract individual time series specific features, enabling the model to specialize to varying time series patterns. Without this component, the overall model would not be able to take a horizon into account that exceeds the maximum feature lag.
This MLP translates one-hot encoding predictors to a fixed length numeric embedding, which is used as additional input on top of the features. The weights of the embedding can only be changed by the backpropagation of the continuous model. Influence of the zero model on the embedding is prevented by introducing a stop gradient.
* <b>Zero model MLP</b>: Predicts the probability of the values being zero (0/1).
* <b>Continuous model MLP</b>: Predicts the continuous targets (the change). The last layer weights of the change model are initialized near zero. Close to persistence as starting point instead of random change predictions.
* <b>Optimizer</b>: One shared Adam optimizer is used for all models.

One of the major difficulties of this competition lies in the number of zeros in the data. This was captured by dividing the prediction in two sub-predictions/models.
* A zero-model, predicting the probability<sup>*</sup> of the next value being a zero or not (when the current value is not missing) 
  * <b> 12<sup>**</sup> binary targets  </b>
* A Continuous model, predicting the real value (when the current value is not zero or missing). This model predicts the change in normalized value from the last value. The predictions are converted using the training range in the post-processing phase.
  * <b>12<sup>**</sup> continuous targets</b>
  
<sub> (*) probabilities were clipped in a post-processing phase: >0.99~1, <0.01~0 </sub>

<sub> (**) depends on the prediction horizon </sub>

To get to the final prediction, the targets of the two models are multiplied as follows: 

<p style="text-align:center;"><b> Final prediction = (1 - probability being zero) * continuous prediction </b></p> 

The cost is defined as follows:

<p style="text-align:center;"><b> Total cost = mse continuous model + 0.05 * cross entropy zero model + 1e-3
* L2 variable norm </b> </p> 

## <a name="closingRemarks"><a> Closing Remarks

### What can be transferred (reused) to other similar challenges?
The <b>usage of neural networks (NN)</b> can be transferred to other challenges. Neural networks are the current state of the art when the problem involves a large amount of data, which is obviously the case here. Furthermore, NN allow maximum flexibility in defining the solution to very specific problems.

The <b>usage of inductive biases</b>, which corresponds to the usage of privileged information that eases the heavy lifting required by the model. These inductive biases are stressed throughout the blog post, but hereby a little overview:
*	Visual inspection of the data cleaning is crucial for the success of the approach.
* The lags are concentrated at a lag of one day
*	Rescale time series so one shared model can be used
*	Predict differences (change) instead of absolute values
*	Combination of zero and change model
*	Make use of a global (zero and change model) and local component (embedding) in model
*	Usage of persistence for time series that enter an unseen range in the adapt phase.
*	Initialize the last layer weights of the change model near zero. This corresponds with predictions close to persistence as the starting point instead of random change predictions.

### How would you further improve the model to make it faster, more accurate?
<b>Increase speed</b>: 
*	Pre-processing the timeseries in batch would significantly increase the prediction speed. At the moment the pre-processing is done one by one.
* I would avoid importing the packages for each prediction round, as this took up one of the three predictions seconds.
* I would optimize the compute time of the feature <i>Time since last non NA</i>, by programming a custom made library using C(++).
* Save time in the pre-processing by excluding time series with very little variation. 
* I would ensure that the multiprocessing is done on GPU's in the training phase. 


<b>Increase accuracy</b>: 
* No auxiliary sources were checked, incorporating auxiliary data could further improve the model accuracy.
* Get more information related to the test data. To give an example, what is the idea behind the interpolations? Are the interpolations also present in the test data?
* The model could be made more global:
    * pushing it to the extreme - all time series could be predicted at once
    * or a possible middle ground - assign time series to k-clusters and perform batch predictions - this           would result in a group and a global encoding.
* Integrating a feedback loop of earlier model predictions could further improve the model accuracy. The idea is to check how good the predictions were and to switch between models based on the prediction errors (e.g. if the predictions are worse then the persistence, stick with the persistence).
* Now the model is updated at fixed predictions steps. This could be changed to adaptive steps depending on the feedback loop of the prediction errors. Or when new time series enter the measurements, make adaptations to the model. 
* I would do a proper hyperparameter tuning in the cloud. In the final submission conservative settings are used as little is known about the unseen data. 

### In which other fields would you see applications for similar prediction challenge and solutions?
Forecasting has applications in a wide range of fields where estimates of future conditions are useful. 
Example of other fields of applications are supply chain management, economic forecasting, earthquake prediction, egain forecasting, sales forecasting, weather forecasting, flood forecasting, meteorology and many others ... .

### What would you recommend to young data scientist or students who want to be succesful?
I would suggest to start with a study of various data science topics. <a href="https://www.coursera.org/learn/machine-learning" target="_blank">Andrew Ng’s course</a> is an excellent place to start. Getting your hands dirty with appropriate feedback is the next step if you want to get better. <a href="www.kaggle.com" target="_blank">Kaggle</a> is of course an excellent platform to do so.

I look forward to your comments and suggestions.


