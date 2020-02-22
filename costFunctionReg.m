function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
for i=1:m
    Xi=X(i,:); Xit=Xi';
    J=J+y(i)*log(sigmoid(theta'*Xit))+(1-y(i))*log(1-sigmoid(theta'*Xit));
end

theta_int=0;
for j=2:size(theta)
    theta_int=theta_int+theta(j,1)^2;
end
    
J=(-J/m)+(lambda/(2*m))*theta_int;


for j=1:1
    grad_int=0; 
    for i=1:m
        Xi=X(i,:); Xit=Xi';
        grad_int=grad_int+(sigmoid(theta'*Xit)-y(i))*X(i,j);
    end
    grad(j,1)=(grad_int/m);
end

for j=2:size(theta)
    grad_int=0; 
    for i=1:m
        Xi=X(i,:); Xit=Xi';
        grad_int=grad_int+(sigmoid(theta'*Xit)-y(i))*X(i,j);
    end
    grad(j,1)=(grad_int/m)+(lambda/m)*theta(j,1);
end
            


% =============================================================

end
