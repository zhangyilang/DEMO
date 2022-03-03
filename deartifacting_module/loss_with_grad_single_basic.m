function [NRMSE,grad,x_hat,PSNR,SSIM] = loss_with_grad_single_basic(data,net,conf,mask)
%LOSS_WITH_GRAD_SINGLE_BASIC loss and grad for ADMMNet-Basic with quantile 
% loss or huber loss module
    
% initialization
N   = numel(net.layers);
res = struct('x',cell(1,N+1),...
             'dzdx',cell(1,N+1),...
             'dzdw',cell(1,N+1));
res(1).x = data.train;

% forword propagation
stage = 0;
for n = 1:N
    l = net.layers{n};
    switch l.type
        case 'X_org'
            res(n+1).x = xorg(res(n).x, l.weights{1}, l.weights{2});
        case 'Convo'
            res(n+1).x = convo(res(n).x, l.weights{1});
        case 'Non_linorg'
            res(n+1).x = zorg(conf.LinearLabel ,res(n).x , l.weights{1});
        case 'Multi_org'
            res(n+1).x = betaorg(res(n-1).x, res(n).x, l.weights{1});
        case 'Quantile'
            stage      = stage + 1;
            res(n+1).x = quantile_transform_response(res(1).x, res(n-3).x, mask, stage,'gpu',conf.EnableGPU);
        case 'Huber'
            res(n+1).x = huber_transform_response(res(1).x, res(n-3).x, mask,'gpu',conf.EnableGPU);
        case 'X_mid'
            res(n+1).x = xmid(res(n-2).x, res(n-1).x, res(n).x, l.weights{1}, l.weights{2});
        case 'Non_linmid'
            res(n+1).x = zmid(conf.LinearLabel, res(n).x, res(n-3).x, l.weights{1});
        case 'Multi_mid'
            res(n+1).x = betamid(res(n-4).x, res(n-1).x, res(n).x, l.weights{1});
        case 'Multi_final'
            res(n+1).x = betafinal(res(n-4).x, res(n-1).x, res(n).x, l.weights{1});
        case 'loss'
            res(n+1).x = rnnloss(res(n).x, data.label);
        otherwise
            error('No such layers type.');
    end
    
end

if nargout >= 1
    NRMSE = gather(res(end).x);
end

% backpropagation
if nargout == 2
    res(end).dzdx{1} = 1;
    for n = N:-1:1
        l = net.layers{n};
        switch l.type
            case 'X_org'
                [res(n).dzdx{1}, res(n).dzdw{1}, res(n).dzdw{2}] = xorg(res(n).x, l.weights{1}, l.weights{2}, res(n+1).dzdx{1});
            case 'Non_linorg'
                [res(n).dzdx{1}, res(n).dzdw{1}] = zorg(conf.LinearLabel, res(n).x, l.weights{1}, res(n+1).dzdx{3}, res(n+3).dzdx{1});
            case 'Multi_org'
                [res(n).dzdx{2}, res(n).dzdx{3}, res(n).dzdw{1}] = betaorg(res(n-1).x, res(n).x, l.weights{1}, res(n+2).dzdx{2}, res(n+4).dzdx{2}, res(n+5).dzdx{1});
            case 'Multi_mid'
                [res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdx{3}, res(n).dzdw{1}] = betamid(res(n-3).x, res(n-1).x, res(n).x, l.weights{1}, res(n+2).dzdx{2}, res(n+4).dzdx{2}, res(n+5).dzdx{1});
            case 'Convo'
                [res(n).dzdx{1}, res(n).dzdw{1}] = convo(res(n).x, l.weights{1}, res(n+1).dzdx{1}, res(n+2).dzdx{2});
            case 'Non_linmid'
                [res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdw{1}] = zmid(conf.LinearLabel, res(n).x, res(n-3).x, l.weights{1}, res(n+1).dzdx{3}, res(n+3).dzdx{1});
            case 'Multi_final'
                [res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdx{3} ,res(n).dzdw{1}] = betafinal(res(n-3).x , res(n-1).x , res(n).x ,l.weights{1}, res(n+2).dzdx{2});
            case 'Quantile'
            case 'Huber'
            case 'X_mid'
                [res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdw{1}, res(n).dzdw{2}] = xmid(res(n-2).x, res(n-1).x, res(n).x, l.weights{1}, l.weights{2}, res(n+1).dzdx{1});
            case 'loss'
                res(n).dzdx{1} = rnnloss(res(n).x, data.label, res(n+1).dzdx{1});
            otherwise
                error('No such layers type.');
        end
    end


    grad = [];
    for n = 1:N
        if isfield(res(n), 'dzdw')
            for i = 1:length(res(n).dzdw)
                gradwei = res(n).dzdw{i};
                grad = [grad;gradwei(:)];
            end
        end
    end
    grad = real(grad);

elseif nargout == 5
    grad = [];
    x_hat = gather(res(end-1).x);
    x_hat(x_hat < 0) = 0;
    x_hat(x_hat > 1) = 1;
    data.label = gather(data.label);
    PSNR = psnr(x_hat,data.label);
    SSIM = ssim(x_hat,data.label);
else
    error('Invalid number of outputs.')
end

end
