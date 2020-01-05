function [projection, eigenvalues] = myLDA(data, num_principal_components)
    d = size(data,2);
    n = size(data,1);
    y = data(:,d);
    x = data(:,1:d-1);
    c = unique(y);
    nc = length(c);
    
    sw = repelem(0,d-1,d-1);
    sb = repelem(0,d-1,d-1);
    m = repelem(0,nc,d-1);
    n = repelem(0,nc);
    for i = 1:nc
        xi = x(y==c(i),:);
        mi = mean(xi);
        %si = cov(xi);
        si = (xi'-mi')*(xi'-mi')';
        sw = sw + si;
        m(i,:) = mi;
        n(i) = size(xi,1);
    end
    mu = mean(m);
    for i = 1:nc
        sb = sb + (n(i).*(m(i,:)'-m')*(m(i,:)'-m')');
    end
    s = pinv(sw)*sb;
    [V,D] = eig(s);
    [~,ind] = sort(diag(D),'descend');
    Ds = D(ind,ind);
    Vs = V(:,ind);
    projection = Vs(:,1:num_principal_components);
    eigenvalues = diag(Ds);
    eigenvalues = eigenvalues(1:num_principal_components);
end