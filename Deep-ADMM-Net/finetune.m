%% Clear

clear
clc
close all

%% Addpath

addpath(genpath('./Train_LBFGS/'))
addpath('./layersfunction/')
addpath('./util/')
addpath('../deartifacting_module/')

%% Initialization

config;

load('./mask/mask_20.mat','mask')
save('./mask.mat','mask')
mask = logical(ifftshift(mask));
if nnconfig.EnableGPU
    mask = gpuArray(mask);
end

load('./Train_output/net/Basic-radial-0.2-real-brain-S10.mat','net');
net = add_layers_basic(net,'Huber');
wei0 = netTOwei(net);
train_dir = './data/DATA-Pseudo-radial-0.2-real-brain/train/';
single_loss_func = @(data,nn)loss_with_grad_single_basic(data,nn,nnconfig,mask);

%% L-BFGS

total_loss_func = @(w)loss_with_grad_total_basic(w,mask,single_loss_func,train_dir,'Huber',nnconfig.EnableGPU);
%parameters in the L-BFGS algorithm
low = -inf*ones(length(wei0),1);
upp = inf*ones(length(wei0),1);
opts.x0 = double(gather(wei0));
opts.m = 5;
opts.maxIts = 1e4;
opts.maxTotalIts = 1e4;
opts.printEvery = 1;
[wei1, l1, info] = lbfgsb(total_loss_func,low,upp,opts);
wei1=single(wei1);
net1 = weiTOnet(wei1);
fprintf('After finetuning, error is %f.\n', l1);
