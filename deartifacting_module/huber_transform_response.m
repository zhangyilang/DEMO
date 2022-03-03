function pseudo_y = huber_transform_response (y,x_hat,mask,varargin)
%HUBER_TRANSFORM_RESPONSE

p = inputParser;
p.addRequired('y');
p.addRequired('x_hat');
p.addRequired('mask');
p.addOptional('delta',10^1.5);
p.addOptional('gpu',true);
p.parse(y,x_hat,mask,varargin{:});

y_hat = fft2(x_hat);
y = y(mask);
y_hat = y_hat(mask);

% estimate noise density function
h0  = mean(abs(y_hat - y) <= p.Results.delta);

% transform response
hloss_deriv = huber_derivative(y_hat - y,p.Results.delta);
if p.Results.gpu
    pseudo_y = zeros(size(mask),'gpuArray');
else
    pseudo_y = zeros(size(mask));
end
pseudo_y(mask) = y_hat - hloss_deriv / h0;

end
