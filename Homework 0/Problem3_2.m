function [w] = Problem3_2(X,y)
    [m,n]=size(X);
    f=[zeros(n,1);ones(m,1)]; % transform problem into a standard LP
    A1=[X.*repmat(y,1,n),eye(m,m)];
    A2=[zeros(m,n),eye(m,m)];
    A=-[A1;A2];
    b=[-ones(m,1);zeros(m,1)];
    x = linprog(f,A,b);% solve LP
    w=x(1:n);% return varible w
    
    scatter(X(:,1), X(:,2), 30, y, 'filled'); % scatter plot of data points
    hold on;
    colormap jet;
    plot([1, -1], [(-(w(1))/(w(2))), ((w(1))/(w(2)))]); % line with decided w
end