traindata = load('optdigits_train.txt');
valdata = load('optdigits_valid.txt');
testdata = load('optdigits_test.txt');
totaldata = [traindata; valdata];

[a,w,v] = mlptrain(traindata,valdata,18,10);
[z,totalerror] = mlptest(totaldata,w,v);
[eigvaluek1,eigvectork1] = pca(z, 2);
m = mean(z);
z1 = z-m;

PCATwo = (eigvectork1'*z1')';
coltab = [50/255 153/255 153/255; 
          100/255 204/255 255/255;
          150/255 210/255 190/255;
          200/255 170/255 160/255;
          250/255 150/255 140/255;
          150/255 50/255 130/255;
          130/255 100/255 210/255;
          0/255 150/255 100/255;
          255/255 200/255 00/255;
          110/255 250/255 230/255];
RGB = coltab(totaldata(:,end)+1, :);
pointsize = 8;

figure;
PCATwo(PCATwo<0) = -log(-PCATwo(PCATwo<0));
PCATwo(PCATwo>0) = log(PCATwo(PCATwo>0));
gscatter(PCATwo(:,1),PCATwo(:,2),totaldata(:,end)+1);
[eigvaluek1,eigvectork1] = pca(z, 3);
m = mean(z);
z1 = z-m;

PCAThree = (eigvectork1'*z1')';

figure;
PCAThree(PCAThree<0) = -log(-PCAThree(PCAThree<0));
PCAThree(PCAThree>0) = log(PCAThree(PCAThree>0));
scatter3((PCAThree(:,1)),(PCAThree(:,2)),(PCAThree(:,3)),pointsize,RGB, 'filled');
text((PCAThree(1:40:end,1)),(PCAThree(1:40:end,2)),(PCAThree(1:40:end,3)),num2str(totaldata(1:40:end,end)),'fontsize',3);