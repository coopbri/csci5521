function [class_pred] = myKNN(training_data, test_data, k)
    x_test = test_data(:,1:size(test_data,2)-1);
    x_train = training_data(:,1:size(training_data,2)-1);
    y_train = training_data(:,size(training_data,2));
    
    distance_matrix = pdist2(x_test, x_train);
    sorted_distance_matrix = distance_matrix;
    max_indices = zeros(size(test_data,1),size(x_train,1));
    
    for i = 1:size(x_test,1)
        [sorted_distance_matrix(i,:),max_indices(i,:)] = sort(distance_matrix(i,:));
    end

    top_predictions = y_train(max_indices(:,1:k));
    
    % Return class prediction
    class_pred = mode(top_predictions,2);
end

