function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
size_g=size(g);
n_lines=size_g(1,1); n_rows=size_g(1,2);
for j=1:n_rows
    for i=1:n_lines
        g(i,j)=1/(1+exp(-1*z(i,j)));
    end
end

% =============================================================

end
