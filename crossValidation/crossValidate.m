function [trainFoldResult, validationFoldResult] = crossValidate(trainInputs, trainOutputs, w, ...
    w_in, leakage, nForgetPoints, reg, k, trainTrueLabel, nOutputUnits, trainingEBIndices, trainingNEBIndices, trainingIndices, videoIndex2sliceIndex, nrSubintervals)
% this script splits the whole data set into k folds and
% we use a stratefied approach
% report the training and testing error along with their virsus
% validationErrorRateVec and trainErrorRateVec will just be a vector of size k
% M, w_out, x will be the optimal parameters taken from the best fold
%%
rng('default');
trainEBlen = length(trainingEBIndices == 1);
trainNEBlen = length(trainingNEBIndices == 2);
indices_one = crossvalind('Kfold', trainEBlen, k);
indices_two = crossvalind('Kfold', trainNEBlen, k);
indices = [indices_one;indices_two]; % each index represent which ford the training video will be assigned to.




%trainErrorRateVec = zeros(k,1);
%validationErrorRateVec = zeros(k,1);


trainFoldResult = cell(k,1);
validationFoldResult = cell(k,1);

%%


for countFold = 1:k % the i-th fold
    %%
    
    validationFoldIndicators = (indices == countFold); % the indicator for the validation fold in training set.
    trainFoldIndicators = ~validationFoldIndicators; % the indicators of the testing fold in trainin set.
    
    
    
    trainFoldVideosIndices = find(trainFoldIndicators);
    validationFoldVideosIndices = find(validationFoldIndicators);
    
    
    trainFoldOriginIndicesAllVideo = trainingIndices(trainFoldVideosIndices);
    
    validationFoldOriginIndicesAllVideo = trainingIndices(validationFoldVideosIndices);
    
    
    trainClipsIndex = cell2mat(videoIndex2sliceIndex(trainFoldOriginIndicesAllVideo));
    validationClipsIndex = cell2mat(videoIndex2sliceIndex(validationFoldOriginIndicesAllVideo));
    
    
    
    
    trainFoldInputs = trainInputs(trainClipsIndex); % record all inputs in train folds in cells
    
    validationFoldInputs = trainInputs(validationClipsIndex); % record all inputs in validation folds in cells
    
    
    
    trainFoldTrueLabel = trainTrueLabel(trainClipsIndex);
    validationFoldTrueLabel = trainTrueLabel(validationClipsIndex);
    
    
    
    yTrainFold = trainOutputs(trainClipsIndex); % record all outputs in train folds in cells
    yValidationFold = trainOutputs(validationClipsIndex); % record all outputs in validation folds in cells
    
    
    intervalsTrainFold = cell(1,1); % interval of trainFolds.
    intervalStartIndex = 1;
    intervalEndIndex = size(yTrainFold{1},1);
    
    
    for i = 1:length(yTrainFold)
        intervalsTrainFold{i,1} = [intervalStartIndex, intervalEndIndex];
        
        intervalStartIndex = intervalEndIndex + 1;
        
        intervalEndIndex = intervalEndIndex + size(yTrainFold{i},1);
        
    end
    intervalsTrainFold = cell2mat(intervalsTrainFold);
    
    
    
    intervalsValidationFold = cell(1,1);
    intervalStartIndex = 1;
    intervalEndIndex = size(yValidationFold{1},1);
    
    
    for i = 1:length(yValidationFold)
        intervalsValidationFold{i,1} = [intervalStartIndex, intervalEndIndex];
        
        intervalStartIndex = intervalEndIndex + 1;
        
        intervalEndIndex = intervalEndIndex + size(yValidationFold{i},1);
        
        
    end
    
    intervalsValidationFold = cell2mat(intervalsValidationFold);
    
    UtrainFold = cell2mat(trainFoldInputs)'; % conver the cells into matrix
    UvalidationFold = cell2mat(validationFoldInputs)';
    
    yTrainFold =  cell2mat(yTrainFold)';
    
    
    %%
    %[M, w_out] = trainESN_EB_temporal(UtrainFold, yTrainFold, w, w_in, leakage, nForgetPoints, intervalsTrainFold, reg);
    [~, w_out, ~] = trainESN_EB(UtrainFold, yTrainFold, w, w_in, leakage, nForgetPoints, intervalsTrainFold, reg, nrSubintervals);
    
    % results on trained fold itself
    %predictionsTrainFold = testESN_EB_temporal(UtrainFold,intervalsTrainFold, w_out, w_in, w, leakage, nOutputUnits, nForgetPoints);
    predictionsTrainFold = testESN_EB(UtrainFold,intervalsTrainFold, w_out, w_in, w, leakage, nForgetPoints, nrSubintervals); % test on training data.

    % results on the validation folds
    %predictionsValidationFold = testESN_EB_temporal(UvalidationFold,intervalsValidationFold, w_out, w_in, w, leakage, nOutputUnits, nForgetPoints);
    predictionsValidationFold = testESN_EB(UvalidationFold,intervalsValidationFold, w_out, w_in, w, leakage, nForgetPoints, nrSubintervals); % test on testing data.
    
    





    
    
    [~, predictionsTrainFold_Labels] = max(predictionsTrainFold,[],2);
    errorTrainFold = sum(trainFoldTrueLabel ~= predictionsTrainFold_Labels);
    
    %errorRateTrainFold = errorTrainFold./size(intervalsTrainFold, 1);
    
    
    
    [~, predictionsValidationFold_Labels] = max(predictionsValidationFold,[],2);
    errorValidationFold = sum(validationFoldTrueLabel ~= predictionsValidationFold_Labels);
    %errorRateValidationFold = errorValidationFold./size(intervalsValidationFold, 1);
    
    
    %trainErrorRateVec(countFold) = errorRateTrainFold;
    %validationErrorRateVec(countFold) = errorRateValidationFold;
    
    
    % Form a confusion matrix for training data

trainTrueNEBindices = find(trainFoldTrueLabel == 1); % indices of genuine training NEB in the data set.
trainEstimateNEBindices = find(predictionsTrainFold_Labels == 1); % indices of estimated NEB in the data set.
trainTrueEBindices = find(trainFoldTrueLabel == 2); % indices of genuine training EB in the data set.
trainEstimateEBindices = find(predictionsTrainFold_Labels == 2); % indices of estimated EB in the data set.

trainTruePositive = length(intersect(trainTrueEBindices,trainEstimateEBindices)); % number of True Positive in classifications in training set
trainFalseNegative = length(intersect(trainTrueEBindices,trainEstimateNEBindices)); % number of False Negative classifications in training set

trainFalsePositive = length(intersect(trainTrueNEBindices,trainEstimateEBindices)); % number of False Positive classifications in training set

trainTrueNegative = length(intersect(trainTrueNEBindices,trainEstimateNEBindices)); % number of True Negative classifications in training set

trainConfusionMatrix = [trainTruePositive,trainFalseNegative;trainFalsePositive,trainTrueNegative ]; % Form a confusion matrix for classification result on training set

F1Train = 2 * trainTruePositive/(length(trainFoldTrueLabel) + trainTruePositive -  trainTrueNegative); % calculate the F1 measure for training set

trainAccuracy = (trainTruePositive + trainTrueNegative )/sum(sum(trainConfusionMatrix)); % calculate the classification accuracy for classification on training set

trainPrecision = trainTruePositive/ (trainTruePositive + trainFalsePositive);
trainRecall = trainTruePositive/ (trainTruePositive + trainFalseNegative);
trainFalseNegativeRate =  trainFalseNegative/(trainFalseNegative + trainTruePositive);


trainFoldResult{countFold} = [F1Train,trainAccuracy,trainPrecision,trainRecall, trainFalseNegativeRate ];


% Form a confusion matrix for validation data
%%

testTrueNEBindices = find(validationFoldTrueLabel == 2);
testEstimateNEBindices = find(predictionsValidationFold_Labels == 2);
testTrueEBindices = find(validationFoldTrueLabel == 1);
testEstimateEBindices = find(predictionsValidationFold_Labels == 1);

testTruePositive = length(intersect(testTrueEBindices,testEstimateEBindices));  % number of True Positive in classifications in testing set
testFalseNegative = length(intersect(testTrueEBindices,testEstimateNEBindices));  % number of False Negative classifications in testing set


testFalsePositive = length(intersect(testTrueNEBindices,testEstimateEBindices)); % number of False Positive classifications in testing set

testTrueNegative = length(intersect(testTrueNEBindices,testEstimateNEBindices)); % number of True Negative classifications in testing set

testConfusionMatrix = [testTruePositive,testFalseNegative;testFalsePositive,testTrueNegative ];  % Form a confusion matrix for classification result on testing set


F1Test = 2 * testTruePositive/(length(validationFoldTrueLabel) + testTruePositive -  testTrueNegative); % calculate the F1 measure for testing result
testAccuracy = (testTruePositive + testTrueNegative )/sum(sum(testConfusionMatrix)); % calculate the classification accuracy for  for classification on testing set
testPrecision = testTruePositive/ (testTruePositive + testFalsePositive);
testRecall = testTruePositive/ (testTruePositive + testFalseNegative);
testFalseNegativeRate =  testFalseNegative/(testFalseNegative + testTruePositive);

validationFoldResult{countFold} = [F1Test,testAccuracy,testPrecision,testRecall, testFalseNegativeRate ];




    
    
    
    
    
    
    
end



%AvTrainErrorRateVec =  trainErrorRateVec;

%AvValidationErrorRateVec  = validationErrorRateVec;







end

