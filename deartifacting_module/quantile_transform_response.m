function pseudo_y = quantile_transform_response (y,x_hat,mask,stage,varargin)
%QUANTILE_TRANSFORM_RESPONSE

p = inputParser;
p.addRequired('y');
p.addRequired('x_hat');
p.addRequired('mask');
p.addRequired('stage');
p.addOptional('tau',0.5);
p.addOptional('c0',1e-4);
p.addOptional('gpu',true);
p.parse(y,x_hat,mask,stage,varargin{:});

y_hat = fft2(x_hat);
y = y(mask);
y_hat = y_hat(mask);

% compute bandwidth
n   = sum(mask(:));  % sample size
s   = 0.01 * numel(mask);   % estimated sparsity level of fft(x)
h   = sqrt( s * log(n) / n ) + sqrt( (p.Results.c0 * s^2 * log(n) / n)^(stage+1) / s );

% estimate noise density function
if p.Results.gpu
    f0 = sum(quantile_kernel_function(gather((y - y_hat) / h))) / h / n;
else
    f0 = sum(quantile_kernel_function((y - y_hat) / h)) / h / n;
end

% transform response
q_loss      = (y <= y_hat) - p.Results.tau;
if p.Results.gpu
    pseudo_y = zeros(size(mask),'gpuArray');
else
    pseudo_y = zeros(size(mask));
end
pseudo_y(mask)  = y_hat - q_loss / f0;

end
