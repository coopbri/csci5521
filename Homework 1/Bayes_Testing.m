% CSCI 5521 Homework 1
% Brian Cooper (coope824 | 5310799)

% BAYES_TESTING Tests a model using Bayes Theorem.
%   [in] p1: learned Bernoulli parameters for C1
%   [in] p2: learned Bernoulli parameters for C2
%   [in] pc1: optimal prior for C1
%   [in] pc2: optimal prior for C2
function [] = Bayes_Testing(test_data, p1, p2, pc1, pc2)
    % Variable aliasing for convenience
    x = test_data(:, 1:end-1);
    
    % Extract label column from test set
    xLabels = test_data(:, end);
    
    % Initialize vectors
    v1 = zeros(size(test_data,1),1);
    v2 = zeros(size(test_data,1),1);
    pred = zeros(size(test_data,1),1);

    for i = 1:size(test_data,1)

            % Compute likelihood for C1 and C2
            p0C1 = p1 .^ (1-x(i,:)');
            p1C1 = (1-p1) .^ (x(i,:)');
            p0C2 = p2 .^ (1-x(i,:)');
            p1C2 = (1-p2) .^ (x(i,:)');

            % Calculate determinant
            v1(i) = pc1 * prod(p0C1 .* p1C1);
            v2(i) = pc2 * prod(p0C2 .* p1C2);

            % Predict class based on performance
            if v1(i) > v2(i)
                pred(i) = 1;
            else
                pred(i) = 2;
            end    
    end

    % Calculate and output error
    error = xLabels - pred;
    accuracy = sum(error(:) == 0) / size(error,1);
    fprintf('Error rate using the optimal prior: %.2f%%\n', (1-accuracy)*100);
end