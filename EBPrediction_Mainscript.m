
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Main Script for Engagement Breakdown Prediction %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This script implements the Engagement Breakdown task as documented in the 
% manuscipt: Anonymous authors (for double blind review purpose), Predicting Engagement Breakdown in HRI Using Thin-slices of Facial Expressions,
%2017.

% created and copyright: Anonymous authors, Oct. 2017


clear;
clc;

%%%%%%%%%%%% Pre-processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ebPreprocessing;



%%%%%%%%%%%% Model setup %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nInputUnits = 13; % number of input units (= nr AUs we get from video preprocessing with FACET). fixed at 12 for this task.
nOutputUnits = 2; % number of output units. fix it as 2 as we have a binary classification task.
in_scale = 1; % w_in will be sampled from [-in_scale, in_scale]
bias_scale = 1;


%nForgetPoints = 50;  % "washout period" for reservoir states collection.
%nInternalUnits= 1000;  % number of iniput units.
%spectralRadius = 0.1;  % spectral radius
%reg = 0.1;  % regularization constant for ridge regression.
%leakage = 0.2; % leaky rate of ESN.
%nrSubintervals = 300; % the number of sub-intervals of the reservoir states into. only the arithmetic average of states in this subinterval are maintained.


nForgetPoints = 50;  % "washout period" for reservoir states collection.
nInternalUnits= 1000;  % number of iniput units.
spectralRadius = 0.1;  % spectral radius
reg = 0.1;  % regularization constant for ridge regression.
leakage = 0.2; % leaky rate of ESN.
nrSubintervals = 300; % the number of sub-intervals of the reservoir states into. only the arithmetic average of states in this subinterval are maintained.




trainInputSignals = cell2mat(trainInputs)';
trainOutputSignals =  cell2mat(trainOutputs)';

testInputSignals = cell2mat(testInputs)';
testOututSignals =  cell2mat(testOutputs)';

disp(sprintf('Start training now!'))

disp(sprintf('Training parameters:'))

% some book keeping before training
disp(sprintf('nForgetPoints  %g, nInternalUnits %g, spectralRadius %g, reg %g, leakage %g,  nrSubintervals %g', ...
              nForgetPoints,      nInternalUnits,     spectralRadius,   reg,   leakage,      nrSubintervals));


%%%%%%%%%%%% Generate a Reservoir %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[w_in, w] = genReservoir(nInternalUnits, nInputUnits, spectralRadius, in_scale, bias_scale);




%%%%%%%%%%%% Training the ESN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t = cputime;
[M1, w_out, teacher1] = trainESN_EB(trainInputSignals, trainOutputSignals, w, w_in, leakage, nForgetPoints, intervalsTrain, reg, nrSubintervals);
e = cputime-t


%%
%%%%%%%%%%%% Testing the ESN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


trainPredictions = testESN_EB(trainInputSignals,intervalsTrain, w_out, w_in, w, leakage, nForgetPoints, nrSubintervals); % test on training data.

t = cputime;
testPredictions = testESN_EB(testInputSignals,intervalsTest, w_out, w_in, w, leakage, nForgetPoints, nrSubintervals); % test on testing data.
e = cputime-t




[~,trainresults] = max(trainPredictions'); % training result
[~,testresults] = max(testPredictions'); % testing result



errorTrain = sum(trainresults' ~= trainTrueLabel); % training error
errorTest = sum(testresults' ~= testTrueLabel); % testing error


%%
% Form a confusion matrix for training data

trainTrueNEBindices = find(trainTrueLabel == 1); % indices of genuine training NEB in the data set.
trainEstimateNEBindices = find(trainresults == 1); % indices of estimated NEB in the data set.
trainTrueEBindices = find(trainTrueLabel == 2); % indices of genuine training EB in the data set.
trainEstimateEBindices = find(trainresults == 2); % indices of estimated EB in the data set.

trainTruePositive = length(intersect(trainTrueEBindices,trainEstimateEBindices)); % number of True Positive in classifications in training set
trainFalseNegative = length(intersect(trainTrueEBindices,trainEstimateNEBindices)); % number of False Negative classifications in training set

trainFalsePositive = length(intersect(trainTrueNEBindices,trainEstimateEBindices)); % number of False Positive classifications in training set

trainTrueNegative = length(intersect(trainTrueNEBindices,trainEstimateNEBindices)); % number of True Negative classifications in training set

%%
trainConfusionMatrix = [trainTruePositive,trainFalseNegative;trainFalsePositive,trainTrueNegative ]; % Form a confusion matrix for classification result on training set

F1Train = 2 * trainTruePositive/(nrTrainingSlices + trainTruePositive -  trainTrueNegative); % calculate the F1 measure for training set

trainAccuracy = (trainTruePositive + trainTrueNegative )/sum(sum(trainConfusionMatrix)); % calculate the classification accuracy for classification on training set

trainPrecision = trainTruePositive/ (trainTruePositive + trainFalsePositive);
trainRecall = trainTruePositive/ (trainTruePositive + trainFalseNegative);
trainFalsePositiveRate =  trainFalsePositive/(trainFalsePositive + trainTrueNegative);
trainFalseNegativeRate =  trainFalseNegative/(trainFalseNegative + trainTruePositive);





% Form a confusion matrix for testing data
%%

testTrueNEBindices = find(testTrueLabel == 2);
testEstimateNEBindices = find(testresults == 2);
testTrueEBindices = find(testTrueLabel == 1);
testEstimateEBindices = find(testresults == 1);

testTruePositive = length(intersect(testTrueEBindices,testEstimateEBindices));  % number of True Positive in classifications in testing set
testFalseNegative = length(intersect(testTrueEBindices,testEstimateNEBindices));  % number of False Negative classifications in testing set


testFalsePositive = length(intersect(testTrueNEBindices,testEstimateEBindices)); % number of False Positive classifications in testing set

testTrueNegative = length(intersect(testTrueNEBindices,testEstimateNEBindices)); % number of True Negative classifications in testing set

testConfusionMatrix = [testTruePositive,testFalseNegative;testFalsePositive,testTrueNegative ];  % Form a confusion matrix for classification result on testing set


F1Test = 2 * testTruePositive/(nrTestingSlices + testTruePositive -  testTrueNegative); % calculate the F1 measure for testing result
testAccuracy = (testTruePositive + testTrueNegative )/sum(sum(testConfusionMatrix)); % calculate the classification accuracy for  for classification on testing set
testPrecision = testTruePositive/ (testTruePositive + testFalsePositive);
testRecall = testTruePositive/ (testTruePositive + testFalseNegative);
testFalsePositive =  testFalsePositive/(testFalsePositive + testTrueNegative);
testFalseNegativeRate =  testFalseNegative/(testFalseNegative + testTruePositive);





%%
disp(sprintf('Training complete!'))



disp(sprintf('TRAIN RESULTS: F1Train %g, trainAcc %g, trainPrecision %g, trainRecall %g  trainFalsePositiveRate %g, trainFalseNegativeRate %g', ...
    F1Train, trainAccuracy, trainPrecision, trainRecall, trainFalsePositiveRate, trainFalseNegativeRate));


disp(sprintf('TESTING RESULTS: F1Test %g, testAcc %g, testPrecision %g, testRecall %g  testFalsePositiveRate %g, testFalseNegativeRate %g', ...
    F1Test, testAccuracy, testPrecision, testRecall, testFalseNegativeRate, testFalseNegativeRate));

