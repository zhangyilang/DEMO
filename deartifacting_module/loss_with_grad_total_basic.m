function [loss, grad] = loss_with_grad_total_basic(weights,mask,loss_func,data_dir,layer_type,GPU)

net = weiTOnet(weights);
net = add_layers_basic(net,layer_type);

files = dir(data_dir);
files = {files.name};
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
    loss = loss / TN;
    
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
    loss = gather(loss / num_files);
    grad = gather(grad / num_files);
else
    error('Invalid number of outputs.');
end

end
