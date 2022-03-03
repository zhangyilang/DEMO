%% Clear

clear
clc
close all

%% Initialization

addpath('./layersfunction/')
addpath('./util/')
addpath('../deartifacting_module/')

config;

% load trained network
load('Train_output/net/Basic-radial-0.2-real-brain-S10.mat','net')
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

% valid data
valid_dir = './data/DATA-Pseudo-radial-0.2-real-brain/valid/';
files = dir(valid_dir);
files = {files.name};
files = files(3:end);
num_files = numel(files);

% delta = 10.^(1:0.5:3);
delta = 2.^(2:6);
num_delta = numel(delta);

NRMSE = zeros(1,num_delta);
PSNR = zeros(1,num_delta);
SSIM = zeros(1,num_delta);

%% Test methods under different SNR

y_sparsity = sum(mask(:));

snr = 10^(nnconfig.SNR / 20);
noise_type = nnconfig.NoiseType;
    
for i = 1:num_files
    
    load(fullfile(valid_dir,files{i}),'data')
    data.train = fft2(data.label) .* mask;
    n_std = std(data.train(mask)) / snr;
    if nnconfig.EnableGPU
        data_noise.label = gpuArray(data.label);
    else
        data_noise.label = data.label;
    end

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
        otherwise
            error('Invalid noise type!')
    end

    if nnconfig.EnableGPU
        data_noise.train = zeros(size(data.train),'gpuArray');
        data_noise.train(mask_device) = gpuArray(data.train(mask) + noise);
    else
        data_noise.train = zeros(size(data.train));
        data_noise.train(mask_device) = data.train(mask) + noise;
    end

    for k = 1:num_delta

        nnconfig.Delta = delta(k);

        % Reconstrction by ADMM-Net with Huber loss
        [rec_nrmse, rec_psnr, rec_ssim] = loss_basic_delta(data_noise, net_DEMO, nnconfig, mask_device);
        NRMSE(k) = NRMSE(k) + rec_nrmse;
        PSNR(k) = PSNR(k) + rec_psnr;
        SSIM(k) = SSIM(k) + rec_ssim;

    end
    
end

NRMSE = NRMSE / num_files;
PSNR = PSNR / num_files;
SSIM = SSIM / num_files;

%% Save

%save('results/valid_delta_Basic-radial0.2-brain-S10-10dB.mat','delta','NRMSE','PSNR','SSIM');
