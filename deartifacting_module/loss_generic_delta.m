function [NRMSE,PSNR,SSIM] = loss_generic_delta(data,net,conf,mask)
%LOSS_GENERIC_DELTA loss and grad for ADMMNet-Generic with
% quantile loss or huber loss module

% initialization
if conf.EnableGPU
    LL  = gpuArray(conf.LinearLabel);
else
    LL = conf.LinearLabel;
end

N 	= numel(net.layers);
res = struct(...
    'x',cell(1,N+1),...
    'dzdx',cell(1,N+1),...
    'dzdw',cell(1,N+1));
res(1).x = data.train;

% forword propagation 
stage   = 0;
for i = 1:N
    l = net.layers{i};
    switch  l.type
        case 'Reorg'
            res(i+1).x = xorg(res(i).x, l.weights{1});
        case 'MIN'
            res(i+1).x = Minus(res(i-3).x, res(i).x) ;
        case 'Multi_org'
            res(i+1).x = betaorg(res(i-4).x, res(i).x, l.weights{1});
        case 'Multi_mid'
            res(i+1).x = betamid(res(i-7).x, res(i-5).x, res(i).x, l.weights{1});
        case 'Multi_final'
            res(i+1).x = betafinal(res(i-7).x, res(i-5).x, res(i).x, l.weights{1});
        case 'Quantile_org'
            stage      = stage + 1;
            res(i+1).x = quantile_transform_response(res(1).x, res(i-5).x, mask, stage,'gpu',conf.EnableGPU);
        case 'Quantile'
            stage      = stage + 1;
            res(i+1).x = quantile_transform_response(res(1).x, res(i-6).x, mask, stage,'gpu',conf.EnableGPU);
        case 'Huber_org'
            res(i+1).x = huber_transform_response(res(1).x, res(i-5).x, mask,'gpu',conf.EnableGPU,'delta',conf.Delta);
        case 'Huber'
            res(i+1).x = huber_transform_response(res(1).x, res(i-6).x, mask,'gpu',conf.EnableGPU,'delta',conf.Delta);
        case 'Remid'
            res(i+1).x = xmid(res(i-2).x, res(i-1).x, res(i).x, l.weights{1});
        case 'ADD'
            res(i+1).x = Add(res(i).x, res(i-2).x);
        case 'c_conv'
            res(i+1).x = comconv(res(i).x, l.weights{1}, l.weights{2});
        case 'conv'
            res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                                  'pad', l.pad, ...
                                  'stride', l.stride, ...
                                  'dilate', 1) ;
        case 'Non_linear'
            res(i+1).x = nonlinear(LL, res(i).x, l.weights{1});
        case 'Refinal'
            res(i+1).x = xfinal(res(i-2).x, res(i-1).x, res(i).x, l.weights{1});
        case 'rLoss'
            res(i+1).x = rnnloss(res(i).x, data.label) ;
        otherwise
            error('invalid layer type.');
    end
end

x_hat = gather(res(end-1).x);
x_zf = gather(real(ifft2(data.train)));
Lxx = sum((x_hat-mean(x_hat,'all')).^2,'all');
Lxy = sum((x_hat-mean(x_hat,'all')).*(x_zf-mean(x_zf,'all')),'all');
b1 = Lxy / Lxx;
b0 = mean(x_zf,'all') - b1 * mean(x_hat,'all');
x_hat = b1 * x_hat + b0;
x_hat(x_hat < 0) = 0;
x_hat(x_hat > 1) = 1;
x_hat(x_hat < 0) = 0;
x_hat(x_hat > 1) = 1;
data.label = gather(real(data.label));
NRMSE = norm(x_hat(:) - data.label(:)) / norm(data.label(:));
PSNR = psnr(x_hat, data.label);
SSIM = ssim(x_hat, data.label);

end
