clear all;clc
data=load('optdigits_train.txt');
labelvec=[1:9,0];
cum=1;
mkdir optdigits_train
cd optdigits_train
for i=1:length(labelvec)
    label=labelvec(i);
    folderName=num2str(label);
    mkdir(folderName);
    row=find(data(:,end)==label);
    data_tmp=data(row,1:end-1);
    cd(folderName)
    disp(['generating training data: Class ',num2str(i)]);
    for j=1:length(row)
        image_tmp=data_tmp(j,:);
        image_tmp = reshape(image_tmp,[8,8]);
        fileName=['image',num2str(cum),'.png'];
        imwrite(image_tmp,fileName);
        cum=cum+1;
    end
    cd ..
end
cd ..

data=load('optdigits_valid.txt');
labelvec=[1:9,0];
cum=1;
mkdir optdigits_valid
cd optdigits_valid
for i=1:length(labelvec)
    label=labelvec(i);
    folderName=num2str(label);
    mkdir(folderName);
    row=find(data(:,end)==label);
    data_tmp=data(row,1:end-1);
    cd(folderName)
    disp(['generating validation data: Class ',num2str(i)]);
    for j=1:length(row)
        image_tmp=data_tmp(j,:);
        image_tmp = reshape(image_tmp,[8,8]);
        fileName=['image',num2str(cum),'.png'];
        imwrite(image_tmp,fileName);
        cum=cum+1;
    end
    cd ..
end
cd ..


clear all;clc
data=load('optdigits_test.txt');
labelvec=[1:9,0];
cum=1;
mkdir optdigits_test
cd optdigits_test
for i=1:length(labelvec)
    label=labelvec(i);
    folderName=num2str(label);
    mkdir(folderName);
    row=find(data(:,end)==label);
    data_tmp=data(row,1:end-1);
    cd(folderName)
    disp(['generating testing data: Class ',num2str(i)]);
    for j=1:length(row)
        image_tmp=data_tmp(j,:);
        image_tmp = reshape(image_tmp,[8,8]);
        fileName=['image',num2str(cum),'.png'];
        imwrite(image_tmp,fileName);
        cum=cum+1;
    end
    cd ..
end
cd ..
disp('done !');