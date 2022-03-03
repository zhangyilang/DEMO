%% Clear

clear
clc
close all

%% Addpath

addpath(genpath('./Train_LBFGS/'))
addpath('./layersfunction/')
addpath('./util/')

%% Initialization

config;

load('./mask/mask_20.mat','mask')
save('./mask.mat','mask')
mask = ifftshift(mask);

net = InitNet();
wei0 = netTOwei(net);
train_dir = './data/DATA-Pseudo-radial-0.2-real-brain/train/';

%% L-BFGS

func = @(w)loss_with_gradient_total(w,mask,train_dir,nnconfig);
%parameters in the L-BFGS algorithm
low = -inf*ones(length(wei0),1);
upp = inf*ones(length(wei0),1);
opts.x0 = double(gather(wei0));
opts.m = 5;
opts.maxIts = 7.2e4;
opts.maxTotalIts = 7.2e4;
opts.printEvery = 1;
[wei1, l1, info] = lbfgsb(func, low, upp, opts);
wei1=single(wei1);
net1 = weiTOnet(wei1);
fprintf('After training, error is %f.\n', l1);
