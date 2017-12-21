%%

clear;
clc;


load('RawSensorDataPreprocessing.mat');


smoothRange = 20;


smoothData = smoothen(RawData, smoothRange); 

%[envHigh, ~] = envelope(tempC,16,'peak');



[ProcessedData, ~, ~] = normalizeUEHRICellData(smoothData);

%[ProcessedData, ~, ~] = normalizeInputs(smoothData);


%%
%dataNr = 1;

%for channel = 1:13

%figure(channel)
%plot(1:size(RawData{dataNr}(:,channel),1), RawData{dataNr}(:,channel)')
%hold on
%plot(1:size(smoothData{dataNr}(:,channel),1), smoothData{dataNr}(:,channel)')
%plot(1:size(ProcessedData{dataNr}(:,channel),1), ProcessedData{dataNr}(:,channel)')

%end


dataNr = 1;
l = 100;
plot(1:l,ProcessedData{dataNr}(1:l,1)', 'r','LineWidth', 2); hold on;
plot(1:l,ProcessedData{dataNr}(1:l,2)', 'k','LineWidth', 1);
plot(1:l,ProcessedData{dataNr}(1:l,3)', 'b','LineWidth', 2);
plot(1:l,ProcessedData{dataNr}(1:l,4)', 'g','LineWidth', 1);
plot(1:l,ProcessedData{dataNr}(1:l,5)', 'k:','LineWidth', 2);
plot(1:l,ProcessedData{dataNr}(1:l,6)', 'k:','LineWidth', 1);
plot(1:l,ProcessedData{dataNr}(1:l,7)', 'k*:','LineWidth', 0.5);
plot(1:l,ProcessedData{dataNr}(1:l,8)', 'g','LineWidth', 0.5);
plot(1:l,ProcessedData{dataNr}(1:l,9)', 'k','LineWidth', 0.5);
plot(1:l,ProcessedData{dataNr}(1:l,10)', '--','LineWidth', 2);
plot(1:l,ProcessedData{dataNr}(1:l,11)', 'm--','LineWidth', 1);
plot(1:l,ProcessedData{dataNr}(1:l,12)', 'c--','LineWidth', 0.5);
plot(1:l,ProcessedData{dataNr}(1:l,13)', 'b--','LineWidth', 0.5);
hold off;
set(gca, 'XTickLabel', [])
set(gca, 'YTickLabel', [])

%axis([1 100 0 1]);
%set(gca, 'FontSize', fs);
%%


%trainingEBIndices = [3,4,5,7,8,12,13,15,16,20,32];
%trainingNEBIndices = [1,2,6,9,24];


%testingEBIndices = [10,11,14,17,21,22,26,27,28,29,31];
%testingNEBIndices = [18,19,23,25,30];



EBIndices = [3,4,5,7,8,12,13,15,16,20,32,10,11,14,17,21,22,26,27,28,29,31];
NEBIndices = [1,2,6,9,24,18,19,23,25,30];

rng('default');
randTrainEBs = randsample(length(EBIndices),11);
trainingEBIndices = EBIndices(randTrainEBs);
testingEBIndices = setdiff(EBIndices,trainingEBIndices);




rng('default');
randTrainNEBs = randsample(length(NEBIndices),5);
trainingNEBIndices = NEBIndices(randTrainNEBs);
testingNEBIndices = setdiff(NEBIndices,trainingNEBIndices);




trainingIndices = horzcat(trainingEBIndices,trainingNEBIndices);
testingIndices = horzcat(testingEBIndices,testingNEBIndices);


cardTrainingSet = length(trainingIndices);
cardTestingSet = length(testingIndices);



thinSlicingSpan = 900; % use last 900 frames (= 30 seconds) for EB training.




%% Preprocessing on the Input of training data.

%trainInputs = cell(nrTrainingSlices,1); % training input contains all slices of videos clips in training set.
trainInputs = cell(1,1);

trainSlicesNEBIndicator = cell(1,1); % For each slice, if there is not engagement breakdown, return 1, otherwise return 0.


countTrainingSlice = 0; % count how many slices in all training clips.



videoIndex2sliceIndex = cell(1,1);


for c = 1:cardTrainingSet
    inputData = ProcessedData{trainingIndices(c)};
    
    
    nrThinSlice = floor(size(inputData,1)./thinSlicingSpan); % nr of slice for this clip.
    
    varLengthThinSlicingSpan = thinSlicingSpan;
    
    if nrThinSlice == 0,
        nrThinSlice = 1;
        varLengthThinSlicingSpan = size(inputData,1);
    end
    
    videoIndex2sliceIndex{trainingIndices(c)} = (countTrainingSlice+1:countTrainingSlice+ nrThinSlice);
    
    
    slicesInVideo = [];
    
    
    
    countSliceInClip = 0;
    for thisSlice = nrThinSlice:-1:1, % play the clips in reverse order
        countSliceInClip  = countSliceInClip+1; % count the nr of this slice.
    
        countTrainingSlice = countTrainingSlice + 1;
        
        
        %slicesInVideo(end+1) = countTrainingSlice;
        
        
        NEBInidcator = (c > 11); % if there is not eb return 1 else return 0.
        trainSlicesNEBIndicator{countTrainingSlice,1} = NEBInidcator;
        endTimeForThisSlice = thisSlice*varLengthThinSlicingSpan; % ending time for this slice.
        trainInputs{countTrainingSlice,1} = inputData(endTimeForThisSlice-varLengthThinSlicingSpan+1:endTimeForThisSlice,:);
    end
    
    %videos2Slices{trainingIndices(c)} = trainVi deos2Slices;
    
    
end

nrTrainingSlices = countTrainingSlice;

trainSlicesNEBIndicator = cell2mat(trainSlicesNEBIndicator);

%% Preprocessing on the Input of testing data.


testInputs = cell(1,1);

testSlicesNEBIndicator = cell(1,1); % For each slice, if there is not engagement breakdown, return 1, otherwise return 0.

countTestingSlice = 0; % count how many slices in all testing clips.



for c = 1:cardTestingSet
    inputData = ProcessedData{testingIndices(c)};
    nrThinSlice = floor(size(inputData,1)./thinSlicingSpan); % nr of slice for this clip.
    
    varLengthThinSlicingSpan = thinSlicingSpan;
    
    if nrThinSlice == 0,
        nrThinSlice = 1;
        varLengthThinSlicingSpan = size(inputData,1);
    end
    
    countSliceInClip = 0;
    for thisSlice = nrThinSlice:-1:1, % play the clips in reverse order
        countTestingSlice = countTestingSlice + 1;
        countSliceInClip  = countSliceInClip+1; % count the nr of this slice.
        
        %NEBInidcator = ( (c > 11)*(thisSlice == nrThinSlice)); % if there is no eb, return 1 else return 0.
        %NEBInidcator = (c > 11); % if there is no eb, return 1 else return 0.
        NEBInidcator = (c > 11); % if there is not eb, i.e. return 1 else return 0.
        
        testSlicesNEBIndicator{countTestingSlice,1} = NEBInidcator;
        endTimeForThisSlice = thisSlice*varLengthThinSlicingSpan; % ending time for this slice.
        testInputs{countTestingSlice,1} = inputData(endTimeForThisSlice-varLengthThinSlicingSpan+1:endTimeForThisSlice,:);
    end
end

nrTestingSlices = countTestingSlice;
testSlicesNEBIndicator = cell2mat(testSlicesNEBIndicator);





%% Output of training data.

trainOutputs = cell(nrTrainingSlices,1);


for countTrainingSlice = 1:nrTrainingSlices,
    if trainSlicesNEBIndicator(countTrainingSlice) == 0,
        trainOutputs{countTrainingSlice} = repmat([1,0],size(trainInputs{countTrainingSlice},1),1);
    else
        trainOutputs{countTrainingSlice} = repmat([0,1],size(trainInputs{countTrainingSlice},1),1);
    end
    
end


%% Output of testing data.

testOutputs = cell(nrTestingSlices,1);


for countTestingSlice = 1:nrTestingSlices,
    if testSlicesNEBIndicator(countTestingSlice) == 0,
        testOutputs{countTestingSlice} = repmat([1,0],size(testInputs{countTestingSlice},1),1);
    else
        testOutputs{countTestingSlice} = repmat([0,1],size(testInputs{countTestingSlice},1),1);
    end
    
end

trainTrueLabel = trainSlicesNEBIndicator+ 1;
testTrueLabel = testSlicesNEBIndicator + 1;

trainTrueIndicator = zeros(size(trainTrueLabel,1),2); % EB: [1, 0 ]
testTrueIndicator =  zeros(size(testTrueLabel,1),2); % NEB: [0,1]



for i = 1 : size(trainTrueLabel,1),
    trainTrueIndicator(i, trainTrueLabel(i)) = 1;
end

for i = 1 : size(testTrueLabel,1),
    testTrueIndicator(i, testTrueLabel(i)) = 1;
end


%%

intervalsTrain = cell(1,1);
intervalStartIndex = 1; 
intervalEndIndex = size(trainOutputs{1},1);


for i = 1:length(trainOutputs)
    intervalsTrain{i,1} = [intervalStartIndex, intervalEndIndex];

    intervalStartIndex = intervalEndIndex + 1;
        
    intervalEndIndex = intervalEndIndex + size(trainOutputs{i},1);
    
    
    
end

intervalsTrain = cell2mat(intervalsTrain);

%%


intervalsTest = cell(1,1);
intervalStartIndex = 1; 
intervalEndIndex = size(testOutputs{1},1);


for i = 1:length(testOutputs)
    intervalsTest{i,1} = [intervalStartIndex, intervalEndIndex];

    intervalStartIndex = intervalEndIndex + 1;
        
    intervalEndIndex = intervalEndIndex + size(testOutputs{i},1);
    
    
    
end

intervalsTest = cell2mat(intervalsTest);





