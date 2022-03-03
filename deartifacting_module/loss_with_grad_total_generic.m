function [loss, grad] = loss_with_grad_total_generic(weights,mask,loss_func,data_dir,layer_type,GPU)

net = weiTOnet(weights);
net = add_layers_generic(net,layer_type);

files = {dir(data_dir).name};
files = files(3:end);
num_files = numel(files);

loss = 0;
if nargout == 1
    for i = 1:num_files
        load(fullfile(data_dir,files{i}),'data');
        data.train = fft2(data.label) .* mask;
        if GPU
            data.train = gpuArray(data.train);
            data.label = gpuArray(data.label);
        end
        l = loss_func(data,net);
        loss = loss + l;
    end
    loss = loss / num_files;
    if GPU
        loss = gather(loss);
    end
    
elseif nargout == 2
    grad = 0;
    for i = 1:num_files
        load(fullfile(data_dir,files{i}),'data');
	data.train = fft2(data.label) .* mask;
        if GPU
            data.train = gpuArray(data.train);
            data.label = gpuArray(data.label);
        end
        [l,g] = loss_func(data,net);
        loss = loss + l;
        grad = grad + g;
    end
    loss = loss / num_files;
    grad = grad / num_files;
    
    if GPU
        loss = gather(loss);
        grad = gather(grad);
    end
    
else
    error('Invalid number of outputs.');
end

end
