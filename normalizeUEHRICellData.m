function [normData, shifts, scales] = normalizeUEHRICellData(data)
% Normalizes training input data of UE-HRI data set. Returns shifts and
% scales of first 12 columns as row vectors (size 12), where 
% normData = scales * (data + shift).
% The last two columns of data are just preserved.


% assemble all samples in one big matrix


totalLength = 0;

for s = 1:size(data,2)
    totalLength = totalLength + size(data{s},1);
end

dimData = size(data{1},2);

allData = zeros(totalLength, dimData);
currentStartIndex = 0;


for s = 1:size(data,2)
    L = size(data{s},1);
    allData(currentStartIndex+1:currentStartIndex+L,:) = ...
        data{s}(:,1:dimData);
    currentStartIndex = currentStartIndex + L;
end

maxVals = max(allData);
minVals = min(allData);
shifts = - minVals;
scales = 1./(maxVals - minVals);
normData = cell(size(data,2),1);
for s = 1:size(data,2)
    normData{s} = data{s}(:,1:dimData) + repmat(shifts, size(data{s},1),1);
    normData{s} = normData{s} * diag(scales);
    % add last two original columns
    %normData{s} = [normData{s} data{s}(:,13:14)];
end


   