% Load image
[img,cmap]= imread('goldy.bmp');
img_rgb =ind2rgb(img,cmap);
img_double=im2double(img_rgb);

% Use k-means to display a compressed image (comp_img)
finalmatrix = reshape(img_double, [],3);
[idx, M] = kmeans(finalmatrix,7); % For goldy test, k = 7

N = size(finalmatrix,1);
comp_img = zeros(N,3);

for j=1 : N
    comp_img(j,:) = M(idx(j),:);
end

comp_img = reshape(comp_img, size(img_rgb,1),size(img_rgb,2),3);        
figure();
imagesc(comp_img);

% Try standard EM algorithm on goldy image
try
    [h_goldy,m_goldy,Q_goldy] = EMG(0,'goldy.bmp',7);
catch
    warning("Must use positive definite covariance matrix for this EMG implementation. Continuing script...");
end

% Run improved EM algorithm on goldy image
[h_goldy_improved,m_goldy_improved,Q_goldy_improved] = EMG(1,'goldy.bmp',7);