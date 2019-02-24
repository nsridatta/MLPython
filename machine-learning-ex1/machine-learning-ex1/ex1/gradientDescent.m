function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %


    h = X * theta; % Hypothesis
    stderr = h - y; % Standard Error
    theta = theta - (alpha/m) * (stderr' * X)'; % Theta (0 and 1)
    J_history(iter) = computeCost(X, y, theta); % Gradient Descent 
    
    %Note while running the prohram in the octave please remove ';' so that the output is printed on the Octave console

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
	disp(min(J_history)); %Display Minimum gradient value or the best fit after computation 
end
