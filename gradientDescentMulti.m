function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    predictions = X * theta;
    
    difference = predictions .- y;

    x1 = X(:,1);
    result1 = sum((difference) .* x1);
    
    x2 = X(:,2);
    result2 = sum((difference) .* x2);
    
    x3 = X(:,3);
    result3 = sum((difference) .* x3);
    
    theta_one = theta(1) - (alpha/m)*sum(result1);
    theta_two = theta(2) - (alpha/m)*sum(result2);
    theta_three = theta(3) - (alpha/m)*sum(result3);
	  theta = [theta_one; theta_two; theta_three];

    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
