function [w_in, w] = genReservoir(nInternalUnits, nInputUnits, spectralRadius, in_scale, bias_scale)
% Generate a reservoir for ESN.

if nInternalUnits <= 20
    connectivity = 1;
else
    connectivity = 10/nInternalUnits;
end



success = 0;

while not(success)
    try
        internalWeights = sprandn(nInternalUnits, nInternalUnits, connectivity);
        opts.disp = 0;
        specRad = abs(eigs(internalWeights,1, 'lm', opts));
        internalWeights = internalWeights/specRad;
        success = 1;
    catch
    end
end


w = spectralRadius*internalWeights;


temp = randn(nInternalUnits, nInputUnits) * diag(in_scale);
bias = bias_scale * randn(nInternalUnits,1);

w_in = [bias, temp];

