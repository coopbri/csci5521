% flag: Binary indicator variable
% image: bitmap image file path
% k: scalar value for number of clusters
function [h,m,q] = EMG(flag,image,k)
    lambda=0.00001;
    if ~exist('flag','var')
        flag = 0;
    end

    % Read image
    [img,cmap] = imread(image);
    img_rgb = ind2rgb(img,cmap);
    img_double = im2double(img_rgb);

    finalmatrix = reshape(img_double,[],3);
    N=size(finalmatrix,1);
    %[idx,m] = kmeans(finalmatrix,k,'EmptyAction','singleton'); % no error version
    [idx,m] = kmeans(finalmatrix,k,'MaxIter',3,'EmptyAction','singleton');
    h = zeros(N,k);
    
    for i=1 : k
        if flag ~= 1
            S(:,:,i) = cov(finalmatrix(idx(:)==i,:));
        else
            S(:,:,i) = cov(finalmatrix(idx(:)==i,:))+lambda*eye(3,3); 
        end
        pi(i) = sum(idx(:)==i)/N;
    end

    q=zeros(150,2);

    for iter=1 : 150
        s1=zeros(N,1);
        for i=1 : k
            p_prev(:,i) = mvnpdf(finalmatrix,m(i,:),S(:,:,i));
            h(:,i) = pi(i)*p_prev(:,i);
            s1(:) = s1(:)+h(:,i);
        end

        h = h ./ s1;
        Ni=sum(h);
        m_updated = h'*finalmatrix;
        m_updated = m_updated./Ni';
        pi_updated = Ni/N;
        sigma_updated=zeros(3,3,k);
        for i=1 : k
            for j=1 : N
                if flag~=1
                    sigma_updated(:,:,i) = sigma_updated(:,:,i)+ h(j,i).*(finalmatrix(j,:)-m(i,:))'*(finalmatrix(j,:)-m(i,:));
                else
                    sigma_updated(:,:,i) = sigma_updated(:,:,i)+ h(j,i).*(finalmatrix(j,:)-m(i,:))'*(finalmatrix(j,:)-m(i,:))+lambda*eye(3,3);
                end
            end
            sigma_updated(:,:,i) = sigma_updated(:,:,i)./Ni(i);
        end

        p_prev(p_prev==0) = 0.00000001;
        pi(pi==0) = 0.00000001; 
        q(iter,1) = sum(h)*transpose(log(pi)) + sum(sum(h.*log(p_prev))); 

        S = sigma_updated;
        m = m_updated;
        pi = pi_updated;

        for i=1 : k
            p_prev(:,i) = mvnpdf(finalmatrix,m(i,:),S(:,:,i));
        end

        p_prev(p_prev==0) = 0.00000001;
        pi(pi==0) = 0.00000001; 
        q(iter,2) = sum(h)*transpose(log(pi))+sum(sum(h.*log(p_prev))); 
    end

    [~, row_def] = max(h, [], 2 );

    comp_img = zeros(N,3);

    for j=1 : N
        comp_img(j,:) = m(row_def(j),:);
    end

    comp_img = reshape(comp_img, size(img_rgb,1), size(img_rgb,2), 3);        
    imagesc(comp_img);
    figure();
    
    x=0.5:0.5:150;
    x1=0.5:1:149.5;
    x2=1:1:150;

    Q=transpose(q);

    hold all;
    scatter(x1,q(:,1),'.','c'); % After E-step
    scatter(x2,q(:,2),'.','m'); % After M-step
    legend('After E-step','After M-Step','AutoUpdate','off');
    plot(x,Q(:)); % Combined smooth line
    xlabel('Iterations');
    ylabel('Log-Likelihood');
    hold off;
end