function net_output = add_layers_generic(net,layer_type,varargin)
%ADD_LAYERS_GENERIC
%   layer_type = 'Quantile' or 'Huber'

p = inputParser;
p.addRequired('net',@(x) isstruct(x) & isfield(net,'layers'));
p.addRequired('type',@(x) strcmp(x,'Quantile') || strcmp(x,'Huber'));
p.addOptional('gpu',true)
p.parse(net,layer_type,varargin{:});
gpu = p.Results.gpu;

net_output.layers = {};
num_layer = 1;
for l = 1:numel(net.layers)
    layer = net.layers{l};
        
    if strcmp(layer.type,'Remid') || strcmp(layer.type,'Refinal')
        net_output.layers{end+1} = struct('type',layer_type, ...
                                    'weights',{{}}, ...
                                    'name',['layer',num2str(num_layer)], ...
                                    'precious',0);
        num_layer = num_layer + 1;
        
    elseif strcmp(layer.type,'conv') || strcmp(layer.type,'c_conv') || strcmp(layer.type,'Non_linear')
        if gpu
            for i = 1:numel(layer.weights)
                layer.weights{i} = gpuArray(layer.weights{i});
                
            end
        end
    end
    
    layer.name = ['layer',num2str(num_layer)];
    num_layer = num_layer + 1;
    net_output.layers{end+1} = layer;
    
end

net_output.layers{7}.type = [layer_type,'_org'];
net_output.meta = net.meta;

end
