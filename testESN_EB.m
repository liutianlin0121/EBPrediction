function predictions = testESN_EB(testInputSignals,intervalsTest, w_out, w_in, w, leakage, nForgetPoints, suppNr)


predictions = zeros(size(intervalsTest, 1), 2);


for countTestSample = 1:size(intervalsTest,1),
    thisTestInput = testInputSignals(:,intervalsTest(countTestSample,1):intervalsTest(countTestSample,2));
    u = thisTestInput(:, 1);
    T = size(thisTestInput, 2);
    x = randn(size(w_in,1),1);
    
    extendedStates = zeros(size([1; u; x],1), T);
    extendedStates(:,1) = [1; u; x];
    
    
    
    
    for countTime=2:T
        % update state
        x = (1 - leakage) * x + leakage * tanh(w_in *[1;u] + w * x);
        %temporalPrediction(:,countTime) = w_out * [1; u; x];
        
        extendedStates(:,countTime) = [1; u; x];
        
        u = thisTestInput(:,countTime);
    end
    
    
        extendedStatesWashedout = extendedStates(:, nForgetPoints + 1 : end);
        weightedExtendedStates = zeros(size(extendedStatesWashedout,1), suppNr);
        
        
     avSpan = floor(size(extendedStatesWashedout,2)/suppNr); % the averaging span.
 
    for s = 1:suppNr
            if s < suppNr
                addState = sum(extendedStatesWashedout(:, (s-1)*avSpan+1:s*avSpan),2)./avSpan;
            else
                addState = sum(extendedStatesWashedout(:, end-avSpan+1:end),2)./avSpan;
            end
            
           weightedExtendedStates(:,s) = addState;
    end
        
        
        
        
            
        
        classHyp = sum(w_out*weightedExtendedStates,2);
                      
        
        if (classHyp(1,:) > classHyp(2,:))
            predictions(countTestSample, :) = [1 0];
        else
            predictions(countTestSample, :) = [0 1];
        end
    
    
end


end







