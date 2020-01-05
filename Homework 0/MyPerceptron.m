% CSCI 5521: Introduction to Machine Learning
% Brian Cooper (coope824)

% MyPerceptron: Basic perceptron implementation
% Arguments:
%   X: Feature matrix
%   y: label vector
%   w0: initial value of weight vector
% Outputs:
%   w: weight vector returned after learning the data
%   step: number of iterations the function took to converge

function [w,step] = MyPerceptron(X, y, w0)
% Plot initial line with data
figure(1);
title("Initial data");
hold on;
    colormap jet; % built-in MATLAB color profile with blue and red on extrema
    scatter(X(:,1), X(:,2), 30, y, 'filled'); % scatter plot of data points
    plot([1, -1], [(-(w0(1))/(w0(2))), ((w0(1))/(w0(2)))]); % initial line
    axis([-1 1 -1 1]);
hold off;

% Setup variables for perceptron
w = w0;                     % initialize w to parameter w0
cur_w = w0 + ones(size(w)); % current w
step = 0;                   % initialize number of steps to 0
n = size(X,1);              % calculate number of training examples

% Run perceptron algorithm
while ~isequal(w, cur_w)
    cur_w = w;
    for i = 1 : n
        if y(i) * (X(i,:) * w) <= 0
            step = step + 1;
            w = w + y(i) .* X(i,:)';
            figure(2);
            clf;
            title("After Perceptron Convergence");
            hold on;
                colormap jet; % built-in MATLAB color profile with blue and red on extrema
                scatter(X(:,1), X(:,2), 30, y, 'filled'); % scatter plot of data points
                plot([1, -1], [(-(w(1))/(w(2))), ((w(1))/(w(2)))]); % line with decided w
                axis([-1 1 -1 1]);
                pause(.9);
            hold off;
        end
    end
end

% Uncomment to only display final convergence (rather than all iterations)
% Plot line after perceptron convergence with data
% figure(2);
% clf;
% title("After Perceptron Convergence");
% hold on;
%     colormap jet; % built-in MATLAB color profile with blue and red on extrema
%     scatter(X(:,1), X(:,2), 30, y, 'filled'); % scatter plot of data points
%     plot([1, -1], [(-(w(1))/(w(2))), ((w(1))/(w(2)))]); % line with decided w
%     axis([-1 1 -1 1]);
% hold off;
end

