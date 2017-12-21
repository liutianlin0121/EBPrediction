function [states, w_out, teachers] = trainESN_EB(trainInputSignals, trainOutputSignals, w, w_in, leakage, nForgetPoints, intervalsTrain, reg, suppNr)
                                   
% collect responses for the internal dynamics 
% u: input signals
% y: teacher signals
% x: internal units a vector of length NX
% w: weight matrix NX by NX
% w_in: input to internal units weights
% w_out: output weight matrix NC by (1 + LP + NX)
% leakage: leaky rate
% intervalsTrain: the range of each data entries
% reg: regulization term


rng('default');
x = randn(size(w,1),1); % initial state of zero. 




avextendedStatesWashedoutOverSamples  = cell(1,1);
avTeacherSignalsOverSamples = cell(1,1);

extendedStateLength = size(x, 1) + size(trainInputSignals, 1) + 1;

for i = 1:size(intervalsTrain, 1)
    extendedStatesWashedout = zeros(extendedStateLength , (intervalsTrain(i,2) - intervalsTrain(i,1)+1 - nForgetPoints) ); 
    teacherSignals = zeros(size(trainOutputSignals, 1), (intervalsTrain(i,2) - intervalsTrain(i,1) + 1 - nForgetPoints));

    avSpan = floor((size(extendedStatesWashedout,2) )/suppNr); % the averaging span.
    weightedextendedStatesWashedout = zeros(extendedStateLength ,suppNr);
    
    
    
    countTime = 1;
            
    
    for timeNow = intervalsTrain(i,1):intervalsTrain(i, 2)
        % update state
        u = trainInputSignals(:, timeNow);

        internal = w * x;
        inputs = w_in * [1; u];
        extendedStatesWashedout_temp = tanh(internal + inputs);
        x = (1 - leakage) * x + leakage * extendedStatesWashedout_temp;

        % discard for the init phase for every sequence
        if countTime > nForgetPoints
            extendedStatesWashedout(:, countTime - nForgetPoints) = [1; u; x];
            teacherSignals(:, countTime - nForgetPoints) = trainOutputSignals(:, timeNow);
        end  
        countTime = countTime + 1;
    end
        
    %avextendedStatesWashedoutOverSamples(:, i) =  sum(extendedStatesWashedout, 2)./size(extendedStatesWashedout,2);
    %avTeacherSignals(:, i) =  sum(teacherSignals, 2)./size(teacherSignals,2);    
    

    avTeacherSignals = repmat(teacherSignals(:,1), [1,suppNr]);

    
    for s = 1:suppNr
            if s < suppNr
                addState = sum(extendedStatesWashedout(:, (s-1)*avSpan+1:s*avSpan),2)./avSpan;
            else
                addState = sum(extendedStatesWashedout(:, end-avSpan+1:end),2)./avSpan;
            end
            
           weightedextendedStatesWashedout(:,s) = addState;
    end
     
    avextendedStatesWashedoutOverSamples{1,i} = weightedextendedStatesWashedout;
    avTeacherSignalsOverSamples{1,i} = avTeacherSignals;

    
end


states = cell2mat(avextendedStatesWashedoutOverSamples);
statesTranspose = states';

teachers = cell2mat(avTeacherSignalsOverSamples);


% ridge regression
w_out =  teachers * statesTranspose * inv(states * statesTranspose + reg * eye(extendedStateLength));


end