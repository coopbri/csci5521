% Load image
[img,cmap] = imread('stadium.bmp');
img_rgb =ind2rgb(img,cmap);
img_double=im2double(img_rgb);

% Use k-means to display a compressed image (comp_img)
finalmatrix = reshape(img_double, [],3);
%[idx, M] = kmeans(finalmatrix,4); % For stadium test, k = 4
%[idx, M] = kmeans(finalmatrix,8); % For stadium test, k = 8
[idx, M] = kmeans(finalmatrix,12); % For stadium test, k = 12

N = size(finalmatrix,1);
comp_img = zeros(N,3);

for j=1 : N
    comp_img(j,:) = M(idx(j),:);
end

comp_img = reshape(comp_img, size(img_rgb,1),size(img_rgb,2),3);        
figure(1);
imagesc(comp_img);

%[h_stadium_4,m_stadium_4,Q_stadium_4] = EMG(0,'stadium.bmp',4); % EM algorithm on stadium image, k = 4
%[h_stadium_8,m_stadium_8,Q_stadium_8] = EMG(0,'stadium.bmp',8); % EM algorithm on stadium image, k = 8
[h_stadium_12,m_stadium_12,Q_stadium_12] = EMG(0,'stadium.bmp',12); % EM algorithm on stadium image, k = 12

% Run script for Q2 part c and e (goldy problems)
hw3_Q2_ce;

% Check if matrix is positive definite
%all(eig(S(:,:,4)) > eps)