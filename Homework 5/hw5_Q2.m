% Data generation
rng(1);                             % For reproducibility
r = sqrt(rand(100,1));              % Radius
t = 2*pi*rand(100,1);               % Angle
data1 = [r.*cos(t), r.*sin(t)];     % Points
r2 = sqrt(3*rand(100,1)+2);         % Radius
t2 = 2*pi*rand(100,1);              % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % Points

% Visualize the data
figure;
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
hold on
plot(data2(:,1),data2(:,2),'b.','MarkerSize',15)
ezpolar(@(x)1);ezpolar(@(x)2);
axis equal
hold off

% Combine data and assign class labels
data3 = [data1;data2];
theclass = ones(200,1);
theclass(1:100) = -1;

% ===========
% Question 2a
% ===========
[a, b] = kernPercGD(data3,theclass);
N = size(data3,1);

% Grid step size
d = 0.02;

[x1Grid,x2Grid] = meshgrid(min(data3(:,1)):d:max(data3(:,1)), ...
     min(data3(:,2)):d:max(data3(:,2)));
xGrid = [x1Grid(:),x2Grid(:)]; 
scores = zeros(size(xGrid,1),1);

for i = 1:size(xGrid,1)
    row = xGrid(i,:);
    scores_total = 0;
    
    for j = 1:N
        plot_data = data3(j,:) * row';
        plot_data = plot_data + 1;
        plot_data = plot_data ^ 2;
        scores_total = scores_total + (a(j) * theclass(j) * plot_data);
    end
    scores_total = scores_total + b;
    scores(i) = scores_total;
end

figure;
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
hold on
plot(data2(:,1),data2(:,2),'b.','MarkerSize',15)
ezpolar(@(x)1);ezpolar(@(x)2);
axis equal

gscatter(data3(:,1),data3(:,2),theclass,'rb','.');
ezpolar(@(x)1);
contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),[0 0],'k');

train_err_rate = 0;

for i=1:size(data3,1)
    scores_total = 0;    
    row = data3(i,:);
    for j = 1:size(data3,1)
        plot_data = row*data3(j,:)';
        plot_data = plot_data + 1;
        plot_data = plot_data ^ 2;
        scores_total = scores_total + (a(j) * theclass(j) * plot_data);
    end
    
    if sign(scores_total + b) ~= theclass(i)
        train_err_rate = train_err_rate+1;
    end
end

train_err_rate = train_err_rate / size(data3,1);
disp(["Training error rate for data3 data: ", num2str(train_err_rate )]);

% ===========
% Question 2b
% ===========
cl = fitcsvm(data3,theclass,'KernelFunction','polynomial','PolynomialOrder',2,'BoxConstraint',1);

% Predict scores over the grid
d = 0.02;
[~,scores] = predict(cl,xGrid);

% Plot the data and the decision boundary
h(1:2) = gscatter(data3(:,1),data3(:,2),theclass,'rb','.');
hold on
ezpolar(@(x)1);
h(3) = plot(data3(cl.IsSupportVector,1),data3(cl.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'g');
legend(h,{'-1','+1','Support Vectors'});
axis equal
hold off

% ===========
% Question 2c
% ===========
% Classify training data and find training error
optdigits49_train=load('optdigits49_train.txt');
train49 = optdigits49_train(:,1:end-1);
class49 = optdigits49_train(:,end);
[a49, b49] = kernPercGD(train49, class49);
train_err_rate=0;

for i = 1:size(train49,1)
    scores_total = 0;
    row = train49(i,:);
    for j = 1:size(train49,1)
        plot_data = row*train49(j,:)';
        plot_data = plot_data + 1;
        plot_data = plot_data ^ 2;
        scores_total = scores_total + (a49(j) * class49(j) * plot_data);
    end
    
    if sign(scores_total + b49) ~= class49(i)
        train_err_rate = train_err_rate+1;
    end
        
end
train_err_rate = train_err_rate / size(train49,1);
disp(["Training error rate for optdigits49 data: ", num2str(train_err_rate )]);

optdigits49_test = load('optdigits49_test.txt');
test49 = optdigits49_test(:,1:end-1);
testclass49 = optdigits49_test(:,end);
test_error49 = 0;

for t = 1:size(test49,1)
    scores_total = 0;
    
    row = test49(t,:);
    for i = 1:size(train49,1)
        plot_data = row * train49(i,:)';
        plot_data = plot_data + 1;
        plot_data = plot_data ^ 2;
        scores_total = scores_total + (a49(i) * class49(i) * plot_data);
    end
    
    if sign(scores_total + b49) ~= testclass49(t)
        test_error49 = test_error49 + 1;
    end
        
end

test_error49 = test_error49 / size(test49,1);
disp(["Testing error rate for optdigits49 data: ", num2str(test_error49)]);

optdigits79_train=load('optdigits79_train.txt');
train79 = optdigits79_train(:,1:end-1);
class79 = optdigits79_train(:,end);

[a79, b79] = kernPercGD(train79, class79);
training_error79=0;

for i = 1:size(train79,1)
    scores_total = 0;
    row = train79(i,:);
    for j = 1:size(train79,1)
        plot_data = row * train79(j,:)';
        plot_data = plot_data + 1;
        plot_data = plot_data ^ 2;
        scores_total = scores_total + (a79(j) * class79(j) * plot_data);
    end
    
    if sign(scores_total + b79) ~= class79(i)
        training_error79 = training_error79 + 1;
    end
end

training_error79 = training_error79 / size(train79,1);
disp(["Training error rate for optdigits79 data: ", num2str(training_error79 )]);

optdigits79_test=load('optdigits79_test.txt');
test79 = optdigits79_test(:, 1:end-1);
testclass79 = optdigits79_test(:, end);
test_error79 = 0;

for i = 1:size(test79,1)
    scores_total = 0;
    row = test79(i,:);
    for j=1:size(train79,1)
        plot_data = row * train79(j,:)';
        plot_data = plot_data + 1;
        plot_data = plot_data ^ 2;
        scores_total = scores_total + (a79(j) * class79(j) * plot_data);
    end
    
    if sign(scores_total + b79) ~= testclass79(i)
        test_error79 = test_error79 + 1;
    end 
end

test_error79 = test_error79 / size(test79,1);
disp(["Testing error rate for optdigits79 data: ", num2str(test_error79)]);