function smoothData = smoothen(data, framesPerSecond)

% smoothen the input data using a filter.

coeff30frames = ones(1, framesPerSecond)/framesPerSecond;

smoothData = cell(size(data,1),1);


cutOff = 30;


for s = 1:size(data,2)
    channelDim = size(data{s},2);
    channelLength = size(data{s},1) - cutOff;
    denoisedSignal = zeros(channelLength, channelDim);
    for thisChannel = 1:channelDim
        denoisedSignal(:,thisChannel) = (filter(coeff30frames, 1, (data{s}(1:end-cutOff,thisChannel))'))';
    end
    smoothData{s} = denoisedSignal; % remove the last 20 second.
end


  


end