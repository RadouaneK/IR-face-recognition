%% ================ Initialization ================
clear ; close all; clc
%%  ================ Load Image Information  ================
faceDatabase = imageSet('TerravicFacialInfrared','recursive');

%%  ================ Display Faces ================

title('Images of Single Face');
for i = 1:size(faceDatabase,2)
 image(i) = faceDatabase(i).ImageLocation(1);
end
montage(image);
title('Images of all Faces');


%%  ================ Split Database into Training & Test Sets  ================
[training,test] = partition(faceDatabase,[0.8 0.2]);
%% Extract and display Histogram of Oriented Gradient Features for single face 
person = 3;
[hogFeature, visualization]= ...
    extractHOGFeatures(read(training(person),1));
figure;
subplot(2,1,1);imshow(read(training(person),1));title('Input Face');
subplot(2,1,2);plot(visualization);title('HoG Feature');

%%  ================Extracting HOG Features for training set  ================
tic
    X = [];
    y = [];
   
for i=1:30
    parfor j = 1:training(i).Count
        X = [X;extractHOGFeatures(read(training(i),j))];
        y = [y,i];
    end
end
toc

%%   ================ Classifier using support vector machine ================
tic
faceClassifier = fitcecoc(X,y);
toc
%%  ================ Extracting features from test set ================
testFeatures = [];
yTest = [];
for i=1:30
    parfor j = 1:test(i).Count
        testFeatures = [testFeatures;extractHOGFeatures(read(test(i),j))];
        yTest = [yTest; i];
    end
end
%%  ================ Predicting the output Acuracy ================
pred = predict(faceClassifier,testFeatures);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == yTest)) * 100);

