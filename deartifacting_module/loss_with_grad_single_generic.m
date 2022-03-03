function [NRMSE,grad,x_hat,PSNR,SSIM] = loss_with_grad_single_generic(data,net,conf,mask)
%LOSS_WITH_GRAD_SINGLE_GENERIC loss and grad for ADMMNet-Generic with
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
            res(i+1).x = huber_transform_response(res(1).x, res(i-5).x, mask,'gpu',conf.EnableGPU);
        case 'Huber'
            res(i+1).x = huber_transform_response(res(1).x, res(i-6).x, mask,'gpu',conf.EnableGPU);
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

if nargout >= 1
    NRMSE = gather(res(end).x);
end

% backpropagation
if  nargout == 2
    res(end).dzdx{1} = 1;   
    for i = N:-1:1
        l = net.layers{i};
        switch l.type
            case 'rLoss'
                res(i).dzdx{1} = rnnloss(res(i).x, data.label, res(i+1).dzdx) ;
            case 'Refinal'
                [res(i).dzdx{1}, res(i).dzdx{2}, res(i).dzdw{1}] = xfinal(res(i-2).x, res(i-1).x, res(i).x, l.weights{1}, res(i+1).dzdx{1});
            case 'Multi_final'
                [res(i).dzdx{1}, res(i).dzdx{2}, res(i).dzdx{3}, res(i).dzdw{1}] = betafinal(res(i-6).x, res(i-5).x, res(i).x, l.weights{1}, res(i+2).dzdx{2});
            case 'MIN'
                [res(i).dzdx{1}, res(i).dzdx{2}] = Minus(res(i-3).x, res(i).x, res(i+1).dzdx{3}, res(i+3).dzdx{1});
            case 'conv'
                [res(i).dzdx{1}, res(i).dzdw{1},res(i).dzdw{2}] = ...
                vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, res(i+1).dzdx{1}, ...
                'pad', l.pad, ...
                'stride', l.stride, ...
                'dilate', 1);
            case 'c_conv'
                 [res(i).dzdx{1}, res(i).dzdw{1}, res(i).dzdw{2}] = comconv(res(i).x, l.weights{1}, l.weights{2}, res(i+1).dzdx{1});
            case 'Non_linear'
                [res(i).dzdx{1}, res(i).dzdw{1}] = nonlinear(LL, res(i).x, l.weights{1}, res(i+1).dzdx{1});
            case 'ADD'
                [res(i).dzdx{1}, res(i).dzdx{2}] = Add(res(i).x, res(i-2).x, res(i+1).dzdx{1}, res(i+4).dzdx{2});
            case 'Remid'
                [res(i).dzdx{1}, res(i).dzdx{2}, res(i).dzdw{1}] = xmid(res(i-2).x, res(i-1).x, res(i).x, l.weights{1}, res(i+1).dzdx{1}, res(i+6).dzdx{2});
            case 'Huber'
            case 'Huber_org'
            case 'Quantile'
            case 'Quantile_org'
            case 'Multi_mid'
                [res(i).dzdx{1}, res(i).dzdx{2}, res(i).dzdx{3}, res(i).dzdw{1}] = betamid(res(i-7).x, res(i-5).x, res(i).x, l.weights{1}, res(i+2).dzdx{2}, res(i+3).dzdx{2}, res(i+8).dzdx{1});
            case 'Multi_org'
                [res(i).dzdx{1}, res(i).dzdx{3}, res(i).dzdw{1}] = betaorg(res(i-4).x, res(i).x, l.weights{1}, res(i+2).dzdx{2}, res(i+3).dzdx{2}, res(i+8).dzdx{1});
            case 'Reorg'
                [res(i).dzdx, res(i).dzdw{1}] = xorg(res(i).x, l.weights{1}, res(i+1).dzdx{1} ,res(i+4).dzdx{2}, res(i+5).dzdx{1});
            otherwise
                error('No such layers type.');
        end
    end
    
    grad= [];
    for n=1:N
        if isfield(res(n), 'dzdw')
            for d = 1:length(res(n).dzdw)
                gradwei=res(n).dzdw{d};
                if conf.Clipif
                    sum_diff = sum(abs(gradwei(:)));
                    if sum_diff > nnconfig.Theta
                        gradwei = nnconfig.Theta / sum_diff *  gradwei;
                    end
                end
                grad = [grad;gradwei(:)];
            end
        end
    end
    grad = double(grad);

elseif nargout == 3
    grad = [];
    x_hat = gather(res(end-1).x);
    x_hat = x_hat + mean(real(gather(ifft2(data.train(:)))) - x_hat(:));
    NRMSE = norm(x_hat(:) - data.label(:)) / norm(data.label(:));
elseif nargout == 5
    grad = [];
    x_hat = gather(res(end-1).x);
    x_zf = gather(real(ifft2(data.train)));
    Lxx = sum((x_hat-mean(x_hat,'all')).^2,'all');
    Lxy = sum((x_hat-mean(x_hat,'all')).*(x_zf-mean(x_zf,'all')),'all');
    b1 = Lxy / Lxx;
    b0 = mean(x_zf,'all') - b1 * mean(x_hat,'all');
    x_hat = b1 * x_hat + b0;
    x_hat(x_hat < 0) = 0;
    x_hat(x_hat > 1) = 1;
    data.label = gather(real(data.label));
    NRMSE = norm(x_hat(:) - data.label(:)) / norm(data.label(:));
    PSNR = psnr(x_hat, data.label);
    SSIM = ssim(x_hat, data.label);
else
    error('Invalid number of outputs.')
end

end
