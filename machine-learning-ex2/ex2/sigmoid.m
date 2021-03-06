function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

[row, col] = size(z);
g = zeros(row, col);
for r = 1:row
  for c = 1:col
    g(r, c) = 1 / (1 + e^(-z(r, c)));
  end
end


% =============================================================

end
