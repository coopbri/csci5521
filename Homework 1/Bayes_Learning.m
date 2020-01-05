% CSCI 5521 Homework 1
% Brian Cooper (coope824 | 5310799)

% BAYES_LEARNING Learns a model using Bayes Theorem. Calculates MLE.
%   [in] training_data: the training data set for learning a model
%   [in] validation_data: the validation data set to verify the model
%   [out] p1: learned Bernoulli parameters for C1
%   [out] p2: learned Bernoulli parameters for C2
%   [out] pc1: optimal prior for C1
%   [out] pc2: optimal prior for C2
function [p1, p2, pc1, pc2] = Bayes_Learning(training_data, validation_data)
    % Variable aliasing for convenience
    t = training_data(:, 1:end-1);
    v = validation_data(:, 1:end-1);

    % Extract label column from training & validation sets
    tClassLabels = training_data(:, end);
    vClassLabels = validation_data(:, end);

    % Map class labels to 1 and 2 (convenience)
    t1 = (tClassLabels(:) == 1);
    t2 = (tClassLabels(:) == 2);

    % Extract records for class 1 and class 2 into new matrices
    T1 = t(t1, :);
    T2 = t(t2, :);

    % Calculate ratio of 0 records to the total number records
    p1 = (sum((T1(:,:)==0)) / size(T1,1))'; % For class 1
    p2 = (sum((T2(:,:)==0)) / size(T2,1))'; % For class 2

    % Initialize sigma values to be tested
    sigma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6];

    % Compute the prior for each sigma
    tPC1 = 1 - exp(-sigma); % For sigma > 0
    tPC2 = 1 - tPC1;

    % vectors for determinant function and predictions
    v1 = zeros(size(validation_data,1),1);
    v2 = zeros(size(validation_data,1),1);
    sigmaPrediction = zeros(size(validation_data,1),1);
    result = zeros(size(sigma,2),1);

    % Likelihood loop for each sigma value
    for i = 1:size(sigma,2)
        for j = 1:size(validation_data,1)

            % Calculate likelihood for each class
            p0C1 = p1 .^ (1-v(j,:)');
            p1C1 = (1-p1) .^ (v(j,:)');
            p0C2 = p2 .^ (1-v(j,:)');
            p1C2 = (1-p2) .^ (v(j,:)');

            % Calculate determinant
            v1(j) = tPC1(1,i) * prod(p0C1 .* p1C1);
            v2(j) = tPC2(1,i) * prod(p0C2 .* p1C2);

            % Choose class based on performance
            if v1(j) > v2(j)
                sigmaPrediction(j) = 1;
            else
                sigmaPrediction(j) = 2;
            end
        end

        % Sum the instances with zero error
        error = vClassLabels - sigmaPrediction;
        result(i) = sum(error(:) == 0);

        fprintf('For sigma = %d, Number of correct predictions: %d, Error rate: %.2f%%\n',...
            sigma(i), result(i), (1-(result(i))/size(v,1))*100);

        % Reset vectors for next sigma calculation
        v1 = zeros(size(validation_data,1),1);
        v2 = zeros(size(validation_data,1),1);
        sigmaPrediction = zeros(size(validation_data,1),1);
    end

    % Select sigma with minimum error on validation dataset
    [value, index] = max(result);
    fprintf('\nBest performance with sigma = %d, Error rate: %.2f%%\n',...
        sigma(index), (1-(value/size(v,1)))*100);

    % Set optimal priors
    pc1 = tPC1(index);
    pc2 = tPC2(index);
end