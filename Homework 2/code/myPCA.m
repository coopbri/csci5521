function [components, eigenvalues] = myPCA(data, num_principal_components)
    s = cov(data);
    [V,D] = eig(s);
    [d,ind] = sort(diag(D),'descend');
    Vs = V(:,ind);
    
    % Return components and eigenvalues
    components = Vs(:,1:num_principal_components);
    eigenvalues = d(1:num_principal_components);
end