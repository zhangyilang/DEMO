function [NRMSE,PSNR,SSIM] = loss_basic_delta(data,net,conf,mask)
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
            res(n+1).x = huber_transform_response(res(1).x, res(n-3).x, mask,'gpu',conf.EnableGPU,'delta',conf.Delta);
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

NRMSE = gather(res(end).x);
x_hat = gather(res(end-1).x);
x_hat(x_hat < 0) = 0;
x_hat(x_hat > 1) = 1;
data.label = gather(data.label);
PSNR = psnr(x_hat,data.label);
SSIM = ssim(x_hat,data.label);

end
