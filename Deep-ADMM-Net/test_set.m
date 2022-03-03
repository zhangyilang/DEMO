%% Clear

clear
clc
close all

%% Initialization

addpath('./layersfunction/')
addpath('./util/')
addpath('../quantile_huber_module/')

config;

% load trained network
load('./Train_output/net/Basic-radial-0.2-real-brain-S10.mat','net')
net_DEMO = add_layers_basic(net, 'Huber');

% load mask
load('./mask/mask_20.mat','mask')
save('./mask.mat','mask')
mask = logical(ifftshift(mask));
if nnconfig.EnableGPU
    mask_device = gpuArray(mask);
else
    mask_device = mask;
end

% test data
test_dir = './data/DATA-Pseudo-radial-0.2-real-brain/test/';
files = dir(test_dir);
files = {files.name};
files = files(3:end);
num_files = numel(files);

noise_type = nnconfig.NoiseType;
SNR = nnconfig.SNR; % dB
num_snr = numel(SNR);

NRMSE = zeros(2,num_snr,num_files);
PSNR = zeros(2,num_snr,num_files);
SSIM = zeros(2,num_snr,num_files);

%% Test methods under different SNR

y_sparsity = sum(mask(:));

for i = 1:num_files

    load(fullfile(test_dir,files{i}),'data')
    data.train = fft2(data.label) .* mask;
    
    if nnconfig.EnableGPU
        data_noise.label = gpuArray(data.label);
    else
        data_noise.label = data.label;
    end

    for j = 1:num_snr

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
                k0 = floor(ky_len*nnconfig.MotionDelay);
                peri_len = ky_len - 2 * k0 - 1;
                ky_mask = randperm(peri_len,round(peri_len * nnconfig.MotionProb))';
                ky_mask(ky_mask > floor(peri_len)/2) = ky_mask(ky_mask > floor(peri_len)/2) + 2 * k0 + 1;
                ky_dist = randi([-nnconfig.MotionDist,nnconfig.MotionDist],length(ky_mask),1);
                noise = zeros(size(mask));
                noise(ky_mask,:) = data.train(ky_mask,:) .* repmat(exp(2j*pi*ky_mask.*ky_dist/ky_len)-1,[1,size(mask,2)]);
                noise = noise(mask);
            case 'Breathing Motion'
                ky_len = size(mask,1);
                ky_mask = rand(ky_len,1) < nnconfig.MotionProb;
                ky_mask(1:floor(ky_len*nnconfig.MotionDelay)) = 0;
                num_line = sum(ky_mask);
                ky_dist = randi([0,nnconfig.MotionDist],sum(num_line),1);
                noise = zeros(size(mask));
                rand_sin = sin((0.1+rand(num_line,1)*4.9) * 2 * pi .* find(ky_mask) / ky_len + rand(num_line,1) * pi / 4);
                noise(ky_mask,:) = data.train(ky_mask,:) .* repmat(exp(2j*pi*find(ky_mask).*ky_dist/ky_len.*rand_sin)-1,[1,size(mask,2)]);
                noise = noise(mask);
            otherwise
                error('Invalid noise type!')
        end
        
        % noise = 0;
        
        if nnconfig.EnableGPU
            data_noise.train = zeros(size(data.train),'gpuArray');
            data_noise.train(mask_device) = gpuArray(data.train(mask) + noise);
        else
            data_noise.train = zeros(size(data.train));
            data_noise.train(mask_device) = data.train(mask) + noise;
        end

        % Reconstrction by ADMM-Net
        [~, rec_img] = loss_with_gradient_single_before(data_noise, net);
        rec_img = gather(rec_img);
        rec_img(rec_img < 0) = 0;
        rec_img(rec_img > 1) = 1;
        NRMSE(1,j,i) = norm(rec_img(:) - data.label(:)) / norm(data.label(:));
        PSNR(1,j,i) = psnr(rec_img, data.label);
        SSIM(1,j,i) = ssim(rec_img, data.label);

        % Reconstrction by ADMM-Net+DEMO
        [rec_nrmse, ~, ~, rec_psnr, rec_ssim] = loss_with_grad_single_basic(data_noise, net_DEMO, nnconfig, mask_device);
        NRMSE(2,j,i) = rec_nrmse;
        PSNR(2,j,i) = rec_psnr;
        SSIM(2,j,i) = rec_ssim;

    end
end

NRMSE_mean = mean(NRMSE,3);
NRMSE_std = std(NRMSE,0,3);
PSNR_mean = mean(PSNR,3);
PSNR_std = std(PSNR,0,3);
SSIM_mean = mean(SSIM,3);
SSIM_std = std(SSIM,0,3);

%% Save results

%
