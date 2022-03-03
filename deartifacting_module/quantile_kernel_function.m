function y = quantile_kernel_function(x)
%QUANTILE_KERNEL_FUNCTION matrix version

y       = zeros(size(x));
T       = (x > -1) & (x < 1);
x_T     = x(T);
y(T)    = (-315 * x_T.^6 + 735 * x_T.^4 - 525 * x_T.^2 + 105) / 64;

end
