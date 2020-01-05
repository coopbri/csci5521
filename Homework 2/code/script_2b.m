% ----------------------- %
% ----- Question 2b ----- %
% ----------------------- %
training_data = load('optdigits_train.txt');
test_data = load('optdigits_test.txt');

n_test = size(test_data,1);
d_test = size(test_data,2);

c = test_data(:,d_test);

y_train = training_data(:,size(training_data,2));
x_train = training_data(:,1:size(training_data,2)-1);

x_test = test_data(:,1:d_test-1);

[Vs, Ds] = myPCA(x_train, size(x_train,2));
pov = Ds(1);

for i=2:length(Ds)
    pov(i) = pov(i-1) + Ds(i);
end

pov = pov./sum(Ds);
plot(1:length(pov), pov, 'c-');
xlabel('Eigenvector');
ylabel('Proportion of Variance');
k = length(pov);

% Determine K eigenvectors that explain 90%+ of the variance
for i=1:length(Ds)
    if pov(i)>=0.9
        k = i;
        break;
    end
end

fprintf("Minimum K value that explains at least 90%% of variance: %d \n", k);
w = Vs(:,1:k);

mean_train = mean(x_train);
z1 = w'*(x_train' - mean_train');
z_train = z1';

mean_test = mean(x_test);
z2 = w'*(x_test' - mean_test');
z_test = z2';

z_train = [z_train, y_train];
z_test = [z_test, c];

k = [1,3,5,7];

for i = k
    cpred = myKNN(z_train, z_test, i);
    err = sum(~(c==cpred))/n_test;
    fprintf("Testing error rate for k = %d: %f \n", i, err);
end

% ----------------------- %
% ----- Question 2c ----- %
% ----------------------- %
sprintf("\n\n\n");
combined_data = [training_data; test_data];
n = size(combined_data,1);
d = size(combined_data,2);
c_2c = combined_data(:,d);

[Vs, Ds] = myPCA(combined_data(:,1:d-1), 2);
w_2c = Vs;
mean_train_2c = mean(combined_data(:,1:d-1));
z_2c = w_2c'*(combined_data(:,1:d-1)' - mean_train_2c');
z_output = z_2c';

z_output = [z_output c_2c];
color_vector = [1,0,0; 0,1,0; 0,0,1; 1,1,0; 1,0,1; 0,1,1; 0.56,0.22,0.56;
                0.62,0.42,0.52; 0.8,0.68,0; 1,0.73,1];
scatter(z_output(:,1),z_output(:,2),10,color_vector(c_2c+1),'filled','MarkerFaceAlpha',0.6);
text(z_output(1:40:end,1),z_output(1:40:end,2),num2str(c_2c(1:40:end)),'FontSize',12);
grid on;
xlabel('Principal Component 1');
ylabel('Principal Component 2');