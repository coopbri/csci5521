% ----------------------- %
% ----- Question 2a ----- %
% ----------------------- %
training_data = load('optdigits_train.txt');
test_data = load('optdigits_test.txt');

n_test = size(test_data,1);
d_test = size(test_data,2);

k = [1,3,5,7];

c = test_data(:,d_test);
d = size(training_data,2);

% Loop through each value of k
for i = k
    cpred = myKNN(training_data, test_data, i);
    err = sum(~(c==cpred))/n_test;
    fprintf("Error rate for k = %d: %f\n", i, err);
end