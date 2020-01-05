function [ eigenvalues,eigvector] = pca(data, num_principal_components)
    [eigvector,Diag] = eig(cov(data));
    [eigenvalues,ind] = sort(diag(Diag),'descend');
    eigvector = eigvector(:,ind);
    eigvector = eigvector(:,1:num_principal_components);
    eigenvalues = eigenvalues(1:num_principal_components);
end