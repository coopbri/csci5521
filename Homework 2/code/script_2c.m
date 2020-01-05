% ----------------------- %
% ----- Question 2d ----- %
% ----------------------- %
training_data = load('optdigits_train.txt');
test_data = load('optdigits_test.txt');
n_test = size(test_data,1);
d_test = size(test_data,2);
n_train = size(training_data,1);
d_train = size(training_data,2);
c = test_data(:,d_test);
y_train = training_data(:,d_train);
    
L = [2,4,9];
k = [1,3,5];

for i = L
    [Vs, Ds] = myLDA(training_data, i);
    w = Vs;
    z1 = w'*(training_data(:,1:d_train-1))';
    z_train = z1';

    z2 = w'*(test_data(:,1:d_test-1))';
    z_test = z2';
    
    z_train = [z_train, y_train];
    z_test = [z_test, c];
    
    for j = k
        cpred = myKNN(z_train, z_test, j);
        err = sum(~(c==cpred))/n_test;
        fprintf("Error rate for (L = %d), (k = %d): %f \n", i, j, err);
    end
end

% ----------------------- %
% ----- Question 2e ----- %
% ----------------------- %
combined_data = [training_data; test_data];
n = size(combined_data,1);
d = size(combined_data,2);
c_2e = combined_data(:,d);

[Vs, Ds] = myLDA(training_data, 2);
w = Vs;
z3 = w'*(combined_data(:,1:d-1))';
z_2e = z3';

color_vector = [1,0,0; 0,1,0; 0,0,1; 1,1,0; 1,0,1; 0,1,1; 0.56,0.22,0.56;
                0.62,0.42,0.52; 0.8,0.68,0; 1,0.73,1];
scatter(z_2e(:,1),z_2e(:,2),10,color_vector(c_2e+1),'filled','MarkerFaceAlpha',0.6);
text(z_2e(1:40:end,1),z_2e(1:40:end,2),num2str(c_2e(1:40:end)),'FontSize',12);
title('LDA Projections');
grid on;