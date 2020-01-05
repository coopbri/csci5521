classdef myLReLULayer < nnet.layer.Layer
    methods
        function layer = myLReLULayer(~, name)
            % Set layer name
            if nargin == 2
                layer.Name = name;
            end
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer and output the result
            Z = max(0, X) + 0.01 .* min(0, X);
        end
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            % Backward propagate the derivative of the loss function through 
            % the layer 
            dLdX = 0.01 .* dLdZ;
            dLdX(X>0) = dLdZ(X>0);
        end
    end
end