function [pc1, pc2, mu1, mu2, s1, s2] = MultiGaussian(training_data, testing_data, Model)
    % Perform the following for all models
    training_size = size(training_data);
    n_training = training_size(1);
    d_training = training_size(2);
    x_training = training_data(:,1:d_training-1);
    y_training = training_data(:,d_training);
    
    class = unique(y_training);
    c1 = class(1);
    c2 = class(2);
    
    x1 = x_training(y_training==c1,:);
    x2 = x_training(y_training==c2,:);
    
    
    pc1 = sum(y_training==c1) / n_training;
    pc2 = 1 - pc1;
    
    mu1 = mean(x1);
    mu2 = mean(x2);
    
    s1 = cov(x1);
    s2 = cov(x2);
    
    % Perform model-based calculations (models 2 and 3)
    % Model 2 has s1 == s2
    % Model 3 has s1,s2 with identical diagonal entries
    if Model == 2
        s1 = pc1*s1 + pc2*s2;
        s2 = s1;
    elseif Model == 3
        alpha1 = 0;
        alpha2 = 0;
        for i = 1 : n_training
            if y_training(i) == c1
                alpha1 = alpha1 + ((x_training(i,:)'-mu1')'*(x_training(i,:)'-mu1'));
            elseif y_training(i) == c2
                alpha2 = alpha2 + ((x_training(i,:)'-mu2')'*(x_training(i,:)'-mu2'));
            end
        end
        alpha1 = alpha1/((d_training-1)*sum(y_training==c1));
        alpha2 = alpha2/((d_training-1)*sum(y_training==c2));
        s1 = alpha1;
        s2 = alpha2;
    end
% ----------------------------------------------------------------------- %
    % Print results
    fprintf("\nModel %d results:\n============================", Model);
    fprintf("\np(C1) = %f\n", pc1);
    fprintf("\np(C2) = %f\n", pc2);
    fprintf("\nmu1 = \n");
    fprintf([repmat('%f   ',1,size(mu1,2)) '\n'],mu1);
    fprintf("\nmu2 = \n");
    fprintf([repmat('%f   ',1,size(mu2,2)) '\n'],mu2);
    if Model == 3
        fprintf("\nalpha1 = %f\n", s1);
        fprintf("\nalpha2 = %f\n", s2);
    else
        fprintf("\nS1 = \n");
        fprintf([repmat('%f   ',1,size(s1,2)) '\n'],s1);
        fprintf("\nS2 = \n");
        fprintf([repmat('%f   ',1,size(s2,2)) '\n'],s2);
    end
    
    s = size(testing_data);
    n = s(1);
    d = s(2);
    x = testing_data(:,1:d-1);
    
    g1 = repelem(0,n);
    g2 = repelem(0,n);
    
    c = testing_data(:,d);
    class_pred = repelem(0,n); 
    
    vsum = 0;
    for i = 1 : n
        if Model==1 || Model==2
            g1(i) = (-0.5*log(det(s1))) + (-0.5*(x(i,:)'-mu1')'*inv(s1)*(x(i,:)'-mu1')) + log(pc1);
            g2(i) = (-0.5*log(det(s2))) + (-0.5*(x(i,:)'-mu2')'*inv(s2)*(x(i,:)'-mu2')) + log(pc2);
        else % Model 3
            s1 = s1*eye(d-1);
            s2 = s2*eye(d-1);
            g1(i) = (-0.5*log(det(s1))) + (-0.5*(x(i,:)'-mu1')'*inv(s1)*(x(i,:)'-mu1')) + log(pc1);
            g2(i) = (-0.5*log(det(s2))) + (-0.5*(x(i,:)'-mu2')'*inv(s2)*(x(i,:)'-mu2')) + log(pc2);
        end
        
        if g1(i)>g2(i)
            class_pred(i) = 1;
        else
            class_pred(i) = 2;
        end
        
        if class_pred(i) ~= c(i)
            vsum = vsum + 1;
        end
    end
    err = vsum / n;
    fprintf("\nTest set error: %f\n============================", err);
end

