%% PLANT DISEASE DETECTION USING LEAF IMAGES

clc; clear; close all;

%% Step 1: Load the dataset
imds = imageDatastore('leaf_dataset', ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');

% Split data into train and test
[trainImgs, testImgs] = splitEachLabel(imds,0.7,'randomized');

%% Step 2: Resize images
inputSize = [64 64 3];
augTrain = augmentedImageDatastore(inputSize,trainImgs);
augTest  = augmentedImageDatastore(inputSize,testImgs);

%% Step 3: Define CNN architecture
layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(numel(unique(trainImgs.Labels)))
    softmaxLayer
    classificationLayer];

%% Step 4: Training options
options = trainingOptions('sgdm', ...
    'MaxEpochs',10, ...
    'InitialLearnRate',0.01, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Step 5: Train the network
net = trainNetwork(augTrain,layers,options);

%% Step 6: Test the network
YPred = classify(net,augTest);
YTest = testImgs.Labels;

accuracy = mean(YPred == YTest);
fprintf('Test Accuracy: %.2f%%\n',accuracy*100);

%% Step 7: Predict a new image
newImage = imread('new_leaf.jpg'); % your test image
newImageResized = imresize(newImage,inputSize(1:2));
label = classify(net,newImageResized);
imshow(newImage);
title(['Predicted Class: ', char(label)]);