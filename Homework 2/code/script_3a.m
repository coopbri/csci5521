% ----------------------- %
% ----- Question 3a ----- %
% ----------------------- %
training_data = load('face_train_data_960.txt');
test_data = load('face_test_data_960.txt');
combined_data = [training_data; test_data];
n = size(combined_data,1);
d = size(combined_data,2);
c = combined_data(:,d);

[Vs, Ds] = myPCA(combined_data(:,1:d-1), 5);
w = Vs';

% Visualize the first 5 eigenfaces
for i=1:5
    subplot(1,5,i);
    imagesc(reshape(w(i,:),32,30)');
    title("Eigenface " + i);
end