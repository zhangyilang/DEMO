function g = huber_derivative(x,delta)
%HUBER_DERIVATIVE

g = x;
T = abs(x) > delta;
g(T) = delta * sign(g(T));

end
