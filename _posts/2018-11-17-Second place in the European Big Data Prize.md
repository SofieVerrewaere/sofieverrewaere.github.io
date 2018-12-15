---
layout: post
title: "European Commission Horizon 2020 prize on Big data Technologies"
subtitle:   "Second place and €400K award!"
date:       2018-11-17 00:00:00
author:     "Sofie Verrewaere"
header-img: "img/EC/front.jpg"
comments: true
---



## Overview

This blog post will cover all sections to go from the raw data to the final submission. Here's an overview of the different sections. If you want to skip ahead, just click the section title to go there.

* *[Introduction](#introduction)*
* *[Model Approach](#modelApproach)*
   * *[Pre-Processing](#preProcessing)*
   * *[Feature engineering](#featEng)*
   * *[Model Architecture](#modelArchitecture)*
   * *[Conclusion](#conclusion)*
* *[Closing Remarks](#closingRemarks)*

## <a name="introduction"><a> Introduction

First of all I would like to thank the EU for organizing the Big Data Technologies H2020 Prize. It has been an amazing experience! I would also like to congratulate the other winners, José Vilar and Yann-Aël Le Borgne for this remarkable achievement!

### Why this prize?
The context of the Big Data Technologies Horizon Prize can be found on the <a href="http://ec.europa.eu/research/horizonprize/index.cfm?prize=bigdata"><b>competition page</b></a>.

Many issues impacting society such as climate change, overcrowded and polluting transportation systems, wasteful energy consumption, would improve with our ability to examine historical records and predict the evolution of these different developments in our society and our economy.

Access to information in a timely way could have a positive impact on the way we consume energy, the way we organise our transport systems and even the way we run our health and other public services. Predicting consumption patterns must be accurate, and need to be delivered in real-time which will allow for effective action such as the use of energy resources in an optimal fashion. Faster, more accurate and resource-efficient prediction methods will result in more efficient management of sectors of the economy. Indeed today, these predictions are not always accurate and lead to the wasteful consumption of energy resources.

The European Commission has launched the Big Data Technologies Horizon Prize to address this issue. The goal is to develop a new spatiotemporal forecasting method which are able to beat ones currently available.

<a href="http://ec.europa.eu/research/participants/data/ref/h2020/other/prizes/contest_rules/h2020-prizes-rules-big-data_en.pdf"><b>Read the detailed rules of the contest </b></a>

### Challenge?
The challenge is to improve the performance of software for the forecasting of geospatio-temporal data (collections of time-stamped records that are linked to a geospatial location).  The prize rewards a solution which improves existing methods in terms of scalability, accuracy, speed and use of computational resources.

The solutions are ranked based on the accuracy of the prediction, expressed as <b>"root-mean square error" (RMSE) </b> and the speed of delivery of the prediction, low use of computing resources expressed as the <b>"overall elapsed execution time" (OEET)</b>.

A <b>training</b> and <b>validation data set</b> were given (in the glossary of the competition this is reffered to as the adapt data set). These data sets contain electrical flow time series with an interval of 5 minutes. In this challenge the aim is to forecast the flow of the next 60 min (1hour), resulting in a forecast horizon of 12 steps. Auxilary data sets were provided, but I decided not to use these in the final submission as no strong guarantees were given that the auxilary data would be available at the prediction time.

The EU provided the contest platform on which the working software submissions are run against a <b>test data</b>. The testing data used at the contest platform was not accessible to the participants. The contest platform measures the performance (accuracy, speed, resource consumption) of each working software submission and was used for testing, and to score and pre-rank the
participants' working software. Yet another data set, verification data (data from the same process, but at a different time
period) was used for the verification runs and final ranking of the pre-selected applications by the jury.

Prior to the opening of the contest platform, a starting kit was provided. This is a simulator of the contest platform that allows a participant to get familiar with the contest running environment and allows for the testing of the working software against sample datasets, representative to the actual test
dataset. 

One of the difficulties of this challenge is the unkown test distribution, which makes it very difficult to improve the model. As little is know of the unseen test data conservative parameter settings are used in the final submission. Lot's of inductive biases are used to ease the heavy lifting of the required model. These inductive biases are stressed throughout the Model Approach.


## <a name="modelApproach"><a> Model Approach

This section aims to summarize on a high level the modeling strategy of the competition approach. All logic is implemented in <b>Python</b>, which is GPL compatible. The Tensorflow package is used to train and load deep learning models. <b>Tensorflow</b> is used for deep learning in Python. <b>Pandas</b> for tabular data handling and <b>Numpy</b> for numeric computations. <b>Multiprocessing</b> is used to parallelize the computations.

### <a name="preProcessing"><a> Pre-processing

The most important steps of the data preprocessing approach are discussed below. The raw data contains all sorts of unusual time series patterns of which the following three are the most important: <b>Short outlier bursts</b>, <b>Interpolated values</b> and <b>Zero values<b/>. 

#### Handling outliers
Short burst outliers are removed from the training data and are ignored completely since it is likely to hurt the modeling capability. The better fit is expected to outweigh benefits from learning about outlier patterns.
{% include image.html url="/img/EC/remove_outliers.jpg" description="<small>Removing Outliers</small>" %}


#### Handling interpolated values
The interpolated values are an artefact of the preprocessing logic in the starting kit. Therefore I decided to NOT include interpolated data points because of the evaluation metric (squared error in future window). Including interpolated values would encourage the model to learn to continue the interpolation to the next real data point but this next real data point will obviously not be available when considering future data!
{% include image.html url="/img/EC/remove_interpolations.jpg" description="<small>Removing Interpolations</small>" %}

#### Handling zero values
Zero values occur frequently in the training data (about one in three data points) and require a special treatment. The huge amount of zero values is captured by defining two types of targets, targets for regular (non-zero) values and the probability of zero values. The combined forecast is the probability of a non-zero value times the forecasted non-zero value which corresponds to the regression target in expectation. 
{% include image.html url="/img/EC/split_into_2_series.jpg" description="<small>Split time series into a 2 series, one for zero values (0/1), one for regular values[0,1]</small>" %}

#### Handling missing values
Missing values are interpolated before doing the preprocessing, this results in those data points being ignored in the model fitting.

#### Scaling, transform and augment data
Neural networks work better when the inputs are in a fixed range. Techniques like batch-norm handle inputs/internal network covariates that have varying ranges but it is likely to be better if the inputs are normalized. The input series were normalized to [0, 1] after exclusion of the outliers. I also augmented this normalized data by containing lags of the input, changes in input time steps and by adding a binary mask to indicate if the input data point is a missing value.
The way the loss is formulated can heavily impact the performance of the model. Benchmarks showed that predicting the change versus the last non missing value works significantly better than predicting next values. This simple change biases the model to predict no change and reserve modeling capacity to focus on true factors of variation.
Incorporating features of the time of the day also showed to help benchmarks significantly. These include periodical features on the hour, day, week and year. Features are calculated by a polar transformation.
Series that are zero / missing for most of the individual series will be treated differently, this is further discussed in the input of the deep learning model.

#### Input Deep Learning Model
Not all the time series are used to feed the deep learning model. The time series are subdivided in valid and invalid data, based on the number of missing values. Series consisting of more than 90% of missing values in the train phase are considered invalid. However, the status (valid/invalid) can change in the validation phase. Three possible scenarios are considered (displayed as 1, 2 and 3 in following figures). 
{% include image.html url="/img/EC/splitdata.jpg" description="<small>Split time series in valid and invalid data</small>" %}

* <b>Scenario 1</b>: Scenario 1 applies to series which are valid at all times (train and validation phase). The number of missing values are limited and the regime (range in particular) is consistent throughout time. The time series are subjected to two types of manipulations, scaling and differentiating. The scaling parameters (max, min,...) are determined in the training phase and saved in the Cache folder. This is a folder for cache files in the output directory. In the validation phase the scaling parameters are reloaded and used to scale the validation time series.
{% include image.html url="/img/EC/scenario1.jpg" description="<small>Scenario 1</small>" %}

* <b>Scenario 2</b>: Scenario 2 applies to series which are valid in the training phase, but act different in the validation phase (e.g. the ranges (min - max) change). The scaling determined in the training phase is no longer valid. If the scaled series exceed 1.2 or drop below -0.2 the series are considered temporary invalid. During temporary invalidness predictions are persistence.
{% include image.html url="/img/EC/scenario2.jpg" description="<small>Scenario 2</small>" %}

* <b>Scenario 3</b>: Scenario 3 applies to series which are invalid in the training phase, but become active in the validation phase. Scaling parameters are determined in the validation phase. 
{% include image.html url="/img/EC/scenario3.jpg" description="<small>Scenario 3</small>" %}

##### Moving window approach and update of model parameters
In the training phase, all possible data was taken into account. In the validation phase, a moving window approach is applied. Only a part of the historical data is taken into account to perform the prediction. The number of steps and the forecast horizon are predetermined by the EU.
{% include image.html url="/img/EC/moving_window.jpg" description="<small>Moving window approach in validation phase.</small>" %}

The deep learning model can be (but is not in the final submission) (pre-)trained and updated on specific moments in time. All data is used in the training phase, with a limited train window, to train the model and the scaling parameters are determined (cfr. Scenario 1). In the validation phase, the model (and scaling parameters in Scenario 3) get updated, every fixed number of steps.
{% include image.html url="/img/EC/model_update.jpg" description="<small>Data Flow</small>" %}

### <a name="featEng"><a> Feature engineering

The predictors used in the final model are:

* The current value and a flag if it is missing value
* Periodical features of the part of the day (sin/cos projection)
* Periodical features of the part of the week (sin/cos projection)
* Lagged values (1, 2, 3, 4, 5, 6, 7, 10, …, 296): first and last lag are absolute scaled values[0, 1], others are relative changes of the scaled lagged values
* Lagged missing values (1, 2, 3, 4, 5, 6, 7, 10, …, 296): True/False/missing values
* Last not missing value: relative change of the lagged values/missing value
* Number of consecutive zeros: scaled [0,1]:
   * 1: >= 300 consecutive values, 
   * 0: no zeros

### <a name="modelArchitecture"><a> Model Architecture

The deep learning model consist of 3 different mlp’s and one optimizer. 

* <b>Embedding mlp</b>: Aims to incorporate a differentiating between individual time series. Translates one-hot encoding predictors to embedding, which is used as input for the zero model and the continuous model. The weights of the embedding can only be changed by the backpropagation of the continuous model. Influence of the zero model on the embedding is prevented by introducing a stop gradient.
* <b>Zero model mlp</b>: Predicts the probability of the values being zero (0/1).
* <b>Continuous model mlp</b>: Predicts the continuous targets
* <b>Optimizer</b>: One adam optimizer is used for all models

{% include image.html url="/img/EC/model_architecture.jpg" description="<small> Model Architecture </small>" %}

#### Targets

One of the major difficulties of this competition lies in the number of zeros in the data. This was captured by dividing the prediction in two sub-predictions/models.
* A zero-model, predicting the probability<sup>*</sup> of the next value being a zero or not (when the current value is not missing) 
  * Targets: 12<sup>**</sup> binary targets  
* A Continuous model, predicting the real value (when the current value is not zero or missing)
  * Targets: 12<sup>**</sup> continuous targets
  
The get to the final prediction, the targets of the two models are multiplied as follows: 
 
<b> Final prediction = (1 - probability being zero) * continuous prediction </b>
  
<sub> (*) probabilities were clipped: >0.99~1, <0.01~0 </sub>

<sub> (**) depends on the prediction horizon </sub>


#### Loss

The cost is defined as follows:

<b>  Total cost = mse continuous model + fraction x cross entropy zero model + L2 penalty x L2 variable norm </b> 




### <a name="conclusion"><a> Conclusion

## <a name="closingRemarks"><a> Closing Remarks




#### What would you do differently if you had unlimited (or a lot more) computing resourcesavailable for the prediction task?

#### In which otther fields would you see applications for similar prediction challenge and solutions?

#### What would you recommend to young data scientist or students whou want to be succesful?

I look forward to your comments and suggestions.


