%% Training Set and Test 

% This code will take the database of our images, split it into to dataSets and convert it into 4 matrices: 
% Training set: - Xtraining : (m x n) where 'm' is the number of images in
% the training set and 'n' is the number of pixels in each image
%               - ytraining : (m x 1) and it contains the actual outputs
% Test set: - Xtest:(a x n) where 'a' is  is the number of images in the
%  test set
%           - ytest: (a x 1)
%  Notice that (m+a) gives the total number of images in the database

%% Initialization

clear ; close all; clc

%% Load Images Information
faceDatabase = imageSet('TerravicFacialInfrared','recursive');% the name of the images should not contain anyparentheses

%% Converting the images into Matrices
tic
 X = [];
 y = [];
 a = [];
for i=1:2
    parfor j = 1:faceDatabase(i).Count
        a = read(faceDatabase(i),j);
        a = double(a);
        X = [X ; a(:)'];
        y = [y;i];
    end
end
toc
%% Mean Normalisation 
for i =1:size(X,2)
     X(:,i) = (X(:,i) - mean(X(:,i)))./255;
end
%%
a = [X y];
p = randperm(size(X,1));
a = a(p,:);


%% Split the DataBase
training = a(1:320,:); % 80% of the data
test = a(321:end,:);  % 20% of the data
%CrossValidation = a(1628:end,:); % 15% of the data

%% inputs and Outputs
Xtraining = training(:,1:76800);
ytraining = training(:,76801);

Xtest = test(:,1:76800);
ytest = test(:,76801);

%XcrossValidation = CrossValidation(:,1:76800);
%YcrossValidation = CrossValidation(:,76801);
