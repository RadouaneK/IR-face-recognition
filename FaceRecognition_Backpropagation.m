%% Initialization
clear ; close all; clc

%% Setup the parameters 
input_layer_size  = 76800; % 320x240 Input Images of Digits
hidden_layer_size = 100;   % 50 hidden neaurons
num_faces = 18;            % 18 labels, from 1 to 18
num_hidden_layers = 1;     % number of hidden layers
lambda = 2;                % Increasing lambda leads to prevent overfitting

%% =========== Part 1: Loading and Visualizing Data =============
%  We start by first loading and visualizing the dataset. 

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('X.mat');
m = size(X, 1);


fprintf('Program paused. Press enter to continue.\n');
pause;
%% ================ Part 2: Initializing Parameters ================
% randomly Initializing Parameters is a very important step because 
% the neurons may end up calculationg 0 gradient

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% ================ Part 3: Implement Backpropagation ================
% here we implement backpropagation with regularization
% 

fprintf('\nImplementing Backpropagation... \n');
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xtraining, ytraining, lambda);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%% =================== Part 8: Training NN ===================
%

fprintf('\nTraining Neural Network... \n')
%   change the MaxIter to a larger value to see how more training helps. 
options = optimset('MaxIter', 100);
[neural_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain ThetaIN and ThetaOUT and Theta back from neural_params

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.


pred = predict(Theta1, Theta2, Xtest);

fprintf('\nTest Set Accuracy: %f\n', mean(pred == ytest) * 100);
