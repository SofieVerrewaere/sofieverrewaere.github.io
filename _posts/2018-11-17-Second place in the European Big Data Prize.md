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
   * *[Pre-Processing](##preProcessing)*
   * *[Feature engineering](##featEng)*
   * *[Model Architecture](##modelArchitecture)*
   * *[Post-processing](##postProcessing)*
   * *[Conclusion](##conclusion)*
* *[Closing Remarks](#closingRemarks)*

## <a name="introduction"><a> Introduction

**From the competition page**: Many issues impacting society such as climate change, overcrowded and polluting transportation systems, wasteful energy consumption, would improve with our ability to examine historical records and predict the evolution of these different developments in our society and our economy.

Access to information in a timely way could have a positive impact on the way we consume energy, the way we organise our transport systems and even the way we run our health and other public services. Predicting consumption patterns must be accurate, and need to be delivered in real-time which will allow for effective action such as the use of energy resources in an optimal fashion. Faster, more accurate and resource-efficient prediction methods will result in more efficient management of sectors of the economy. Indeed today, these predictions are not always accurate and lead to the wasteful consumption of energy resources.

The European Commission has launched the Big Data Technologies Horizon Prize to address this issue. The goal is to develop a new spatiotemporal forecasting method which are able to beat ones currently available.


## <a name="modelApproach"><a> Model Approach

INTRO

### <a name="preProcessing"><a> Pre-processing

The most important steps of the data preprocessing approach are discussed below.

The raw data contains all sorts of unusual time series patterns of which the following three are the most important:

* Short outlier bursts (1 to 10 data points)
* Zero values (ranging from 1 data point to long sequences of missing (?) data)
* Interpolated values


#### Handling outliers
Short burst outliers were removed from the training data and are ignored completely since it is likely to hurt the modeling capability. The better fit is expected to outweigh benefits from learning about outlier patterns.
{% include image.html url="/img/EC/remove_outliers.jpg" description="Removing Outliers" %}


#### Handling interpolated values
The interpolated values are an artefact of the preprocessing logic in the starting kit. The starting kit contains one observation for each 5-minute time step but the real data is not going to be in this format. The organisers announced that the input data will contain arbitrary time gaps. Therefore I decided to NOT include interpolated data points because of the evaluation metric (squared error in future window). Including interpolated values would encourage the model to learn to continue the interpolation to the next real data point but this next real data point will obviously not be available when considering future data!
{% include image.html url="/img/EC/remove_interpolations.jpg" description="<small>Removing Interpolations<small>" %}

#### Handling zero values
Zero values occur frequently in the training data (about one in three data points) and require a special treatment. The huge amount of zero values was captured by defining two types of targets, targets for regular (non-zero) values and the probability of zero values. The combined forecast is the probability of a non-zero value times the forecasted non-zero value which corresponds to the regression target in expectation. 
{% include image.html url="/img/EC/split_into_2_series.jpg" description="Split time series into a 2 series, one for zero values (0/1), one for regular values[0,1]" %}

#### Handling missings
Missings are interpolated before doing the preprocessing, this will result in those data points being ignored in the model fitting.

#### Scaling, transform and augment data
Neural networks work better when the inputs are in a fixed range. Techniques like batch-norm handle inputs/internal network covariates that have varying ranges but it is likely to be better if we control the normalization of the inputs. The input series were normalized to [0, 1] after exclusion of the outliers. I also augmented this normalized data by containing lags of the input, changes in input time steps and by adding a binary mask to indicate if the input data point was missing.
The way the loss is formulated can heavily impact the performance of the model. Benchmarks showed that predicting the change versus the last non missing value works significantly better than predicting next values. This simple change biases the model to predict no change and reserve modeling capacity to focus on true factors of variation.
Incorporating features of the time of the day also showed to help benchmarks significantly. These include periodical features on the hour, day, week and year. Features are calculated by a polar transformation.
Series that are zero / missing for most of the individual series will be treated differently, this is further discussed in the deep learning model section.

#### Input Deep Learning Model

Not all the time series are used to feed the deep learning model. The time series are subdivided in valid and invalid data, based on the number of missings. Series consisting of more than 90% of missings in the train phase are considered invalid. However, the status (valid/invalid) can change in the adapt phase. Three possible scenarios are considered (displayed as 1, 2 and 3 in following figures). 

![Split data](/img/EC/splitdata.jpg)

* *Scenario 1*: Scenario 1 applies to series which are valid at all times (train and adapt phase). The number of missings are limited and the regime (range in particular) is consistent throughout time. The time series are subjected to two types of manipulations, scaling and differentiating. The scaling parameters (max, min,...) are determined in the training phase and saved in the Cache folder. In the adapt phase the scaling parameters are reloaded and used to scale the adapt time series.

![Scenario 1](/img/EC/scenario1.jpg)

* *Scenario 2*: Scenario 2 applies to series which are valid in the training phase, but act different in the adapt phase (e.g. the ranges (min - max) change). The scaling determined in the training phase is no longer valid. If the scaled series exceed 1.2 or sink below -0.2 the series are considered temporary invalid. During temporary invalidness predictions are persistence.

![Scenario 2](/img/EC/scenario2.jpg)

* *Scenario 3*: Scenario 3 applies to series which are invalid in the training phase, but become active in the adapt phase. Scaling parameters are determined in the adapt phase. 

![Scenario 3](/img/EC/scenario3.jpg)


### <a name="featEng"><a> Feature engineering

The predictors used in the final model are:

* The current value and a flag if it is missing
* Periodical features of the part of the day (sin/cos projection)
* Periodical features of the part of the week (sin/cos projection)
* Lagged values (1, 2, 3, 4, 5, 6, 7, 10, …, 296): first and last lag are absolute scaled values[0, 1], others are relative changes of the scaled lagged values (more detailed info is given below in §input).
* Lagged missings (1, 2, 3, 4, 5, 6, 7, 10, …, 296): True/False/Missings
* Last not missing value: relative change of the lagged values/missing
* Number of consecutive zeros: scaled [0,1], 
   * 1: >= 300 consecutive values, 
   * 0: no zeros



### <a name="modelArchitecture"><a> Model Architecture

The deep learning model consist of 3 different mlp’s and one optimizer. 

* Embedding mlp: Aims to incorporate a differentiating between individual time series. Translates one-hot encoding predictors to embedding, which is used as input for the zero model and the continuous model. The weights of the embedding can only be changed by the backpropagation of the continuous model. Influence of the zero model on the embedding is prevented by introducing a stop gradient.
* Zero model mlp: Predicts the probability of the values being zero (0/1).
* Continuous model mlp: Predicts the continuous targets
* Optimizer: One adam optimizer is used for both models, 
![Model](/img/EC/model_architecture.jpg)

#### Targets

One of the major difficulties of this competition lies in the number of zeros in the data. This was captured by dividing the prediction in two sub-predictions/models.
* A zero-model, predicting the probability* of the next value being a zero or not (when the current value is not missing) 
  * Targets: 12** binary targets  
* A Continuous model, predicting the real value (when the current value is not zero or missing)
  * Targets: 12** continuous targets
  
The get to the final prediction, the targets of the two models are multiplied as follows: 
 
  Final prediction = (1 - probability being zero) * continuous prediction
  
(*) probabilities were clipped: >0.99~1, <0.01~0
(**) depends on the prediction horizon


#### Loss

the cost is defined as follows:
      Total cost: mse continuous model + fraction * cross entropy zero model + L2 penalty*L2 variable norm

### <a name="postProcessing"><a> Post-processing


### <a name="conclusion"><a> Conclusion

## <a name="closingRemarks"><a> Closing Remarks


#### What would you do differently if you had unlimited (or a lot more) computing resourcesavailable for the prediction task?

#### In which otther fields would you see applications for similar prediction challenge and solutions?

#### What would you recommend to young data scientist or students whou want to be succesful?

I look forward to your comments and suggestions.


