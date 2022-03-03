function net_output = add_layers_basic(net,layer_type)
%ADD_LAYERS_BASIC
%   layer_type = 'Quantile' or 'Huber'

p = inputParser;
p.addRequired('net',@(x) isstruct(x) & isfield(net,'layers'));
p.addRequired('type',@(x) strcmp(x,'Quantile') || strcmp(x,'Huber'));
p.parse(net,layer_type);

net_output.layers = {};
for l = 1:numel(net.layers)
    layer = net.layers{l};
    if strcmp(layer.type,'X_mid')
        net_output.layers{end+1}.type = layer_type;
    end
    net_output.layers{end+1} = layer;
end

end
