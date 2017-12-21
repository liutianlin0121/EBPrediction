clear;
clc;

%%%%%%%%%%%% Preprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ebPreprocessing;


% model setup




nInputUnits = 13; % number of input units (= nr AUs we get from video preprocessing with FACET), fixed at 12 for this task.
nOutputUnits = 2; % number of output units. 2 if we have a binary classification
in_scale = 1; % w_in will be sampled from [-in_scale, in_scale]
bias_scale = 1;

k = 2;


%%
nForgetPoints = 50;  % "washout period" for reservoir states collection.
nInternalUnits= 1000;  % number of iniput units.
spectralRadius = 0.1;  % spectral radius
reg = 0.1;  % regularization constant for ridge regression.
leakage = 0.2; % leaky rate of ESN.
nrSubintervals = 300; % the number of sub-intervals of the reservoir states into. only the arithmetic average of states in this subinterval are maintained.




[w_in, w] = genReservoir(nInternalUnits, nInputUnits, spectralRadius, in_scale, bias_scale);

%%
[trainFoldResult, validationFoldResult] = crossValidate(trainInputs, trainOutputs, w, w_in, leakage, nForgetPoints, reg, k, trainTrueLabel, nOutputUnits,trainingEBIndices,trainingNEBIndices, trainingIndices, videoIndex2sliceIndex,nrSubintervals);

%disp(sprintf('trainErr  %g, validationErr %g, , leakage %g, reg %g,  SR %g, N %g, forgetPoints %g', ...
%    mean(AvTrainErrorRateVec), mean(AvValidationErrorRateVec),leakage, reg, spectralRadius, nInternalUnits, nForgetPoints ));



