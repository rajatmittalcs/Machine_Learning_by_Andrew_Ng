function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%C_sigma is the set of values to find the best combo

C_sigma=[0.01,0.03,0.1,0.3,1,3,10,30];
errors=zeros(size(C_sigma,2),size(C_sigma,2));
%errors is a matrix to collect all the error values in below iterations

for i=1:numel(C_sigma)
for j=1:numel(C_sigma)

C=C_sigma(i);
sigma=C_sigma(j);

%Training the model on X and y data with iterations of C and sigma
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

%Predicting values on cross validation set
predictions=svmPredict(model,Xval);

%Caluclating error in above predictions
error=mean(double(predictions ~= yval));

%Adding all the errors in one erros matrix
errors(i,j)=error;


end
end

%Finding the minimum error value
min_error=min(min(errors));

%Finding indexes of the min error value to find the best values of C n sigma
[indX,indY]=find(errors==min_error);
C=C_sigma(indX(1));
sigma=C_sigma(indY(1));
% =========================================================================

end
