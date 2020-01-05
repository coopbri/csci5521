traindata = load('optdigits_train.txt');
valdata   = load('optdigits_valid.txt');
testdata  = load('optdigits_test.txt');

valid = zeros(1,6);
train = zeros(1,6);
test  = zeros(1,6);

k = 1;

for i = 3:3:18
    fprintf("Testing %d hidden units:\n", i)
    [z,w,v]      = mlptrain(traindata,valdata,i,10);
    fprintf("\nDetermining validation set error...\n")
    [~,valid(k)] = mlptest(valdata,w,v);
    
    fprintf("\nDetermining training set error...\n")
    [~,train(k)] = mlptest(traindata,w,v);
    
    fprintf("\nDetermining test set error...\n")
    [~,test(k)]  = mlptest(testdata,w,v);
    k = k+1;
    fprintf("\n===================================\n")
end

i = 3:3:18;
plot(i,valid); % validation error
hold on;
plot(i,train); % training error
xlabel("Number of Hidden Units"); ylabel("Error Rate");
legend("Validation Error", "Training Error");
disp(test);