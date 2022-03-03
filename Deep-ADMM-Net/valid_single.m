%% Clear

clear
clc
close all

%% Initialization

addpath('./layersfunction/')
addpath('./util/')
addpath('../deartifacting_module/')

% load trained network
load('./Train_output/net/Basic-radial-0.2-real-brain-S10.mat','net')
net_DEMO = add_layers_basic(net, 'Huber');

% load mask
load('./mask/mask_20.mat','mask')
save('./mask.mat','mask')
mask = logical(ifftshift(mask));

% load valid data
load('./data/DATA-Pseudo-radial-0.2-real-brain/valid/04.mat','data')
data.train = fft2(data.label) .* mask;

config;
noise_type = nnconfig.NoiseType;

NRMSE = zeros(3,1);
PSNR = zeros(3,1);
SSIM = zeros(3,1);

%% Add noises

y_sparsity = sum(mask(:));

switch noise_type
    case 'Herringbone'
        y_std = std(data.train(mask));
        n_std = y_std / 10^(nnconfig.SNR / 20);
        noise_mask = rand(y_sparsity,1) < nnconfig.OutlierProb;
        num_outliers = sum(noise_mask);
        noise = zeros(y_sparsity,1);
        noise(noise_mask) = randn(num_outliers,1) + 1i * randn(num_outliers,1);
        noise = noise / std(noise) * n_std;
    case 'Motion'
        ky_len = size(mask,1);
        k0 = floor(ky_len * nnconfig.MotionDelay);
        peri_len = ky_len - 2 * k0;
        ky_mask = k0 + randperm(peri_len,round(peri_len * nnconfig.MotionProb))';
        ky_dist = randi([-nnconfig.MotionDist,nnconfig.MotionDist],length(ky_mask),1);
        noise = zeros(size(mask));
        noise(ky_mask,:) = data.train(ky_mask,:) .* repmat(exp(2j*pi*ky_mask.*ky_dist/ky_len)-1,[1,size(mask,2)]);
        noise = noise(mask);
    otherwise
        error('Invalid noise type!')
end

% add noise
data.train(mask) = data.train(mask) + noise;

%% Transfer to GPU

im_ori = data.label;
if nnconfig.EnableGPU
   data.train = gpuArray(data.train);
   data.label = gpuArray(data.label);
   mask = gpuArray(mask);
end

%% Reconstruction by zero-filling

rec_img_zf = gather(real(ifft2(data.train)));
NRMSE(1) = norm(rec_img_zf(:) - im_ori(:)) / norm(im_ori(:));
PSNR(1) = psnr(rec_img_zf, im_ori);
SSIM(1) = ssim(rec_img_zf, im_ori);

%% Reconstrction by ADMM-Net

[NRMSE(2), rec_img_ADMMNet] = loss_with_gradient_single_before(data, net);
rec_img_ADMMNet = gather(rec_img_ADMMNet);
PSNR(2) = psnr(rec_img_ADMMNet, im_ori);
SSIM(2) = ssim(rec_img_ADMMNet, im_ori);

%% Reconstrction by ADMM-Net+DEMO

[NRMSE(3), ~, rec_img_huber, PSNR(3), SSIM(3)] = loss_with_grad_single_basic(data, net_DEMO, nnconfig, mask);

%% Display

% figure(1);

subplot(2,3,1)
imshow(im_ori)
title('Original');

subplot(2,3,4);
imshow(rec_img_zf)
title('Zero-filling');

subplot(2,3,5);
imshow(rec_img_ADMMNet)
title('ADMM-Net');

subplot(2,3,6);
imshow(rec_img_huber)
title('ADMM-Net+DEMO');

subplot(2,3,2)
imshow(3 * abs(rec_img_ADMMNet - data.label))
subplot(2,3,3)
imshow(3 * abs(rec_img_huber - data.label))

Methods = {'Zero-filling';'ADMM-Net';'ADMM-Net+DEMO'};
table(Methods, NRMSE, PSNR, SSIM)
