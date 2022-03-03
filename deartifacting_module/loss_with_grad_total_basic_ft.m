[ loss, grad ] = loss_with_grad_total_basic_ft(weights,data_dir,conf)

net = weiTOnet(weights);

files = dir(data_dir);
files = {files.name};
files = files(3:end);
num_files = numel(files);

loss = 0;
if nargout == 1
    for i = 1:num_files
        load(fullfile(data_dir,files{i}),'data');
%         data.train = fft2(data.label) .* mask;
        if conf.EnableGPU
            data.train = gpuArray(data.train);
            data.label = gpuArray(data.label);
        end
        l = loss_with_gradient_single(data, net, conf);
        loss = loss + l;
    end
    loss = loss / num_files;
    
elseif nargout == 2
    grad_length = length(weights);
    grad = zeros(grad_length,1);
    for i = 1:num_files
        load(fullfile(data_dir,files{i}),'data');
%         data.train = fft2(data.label) .* mask;
        if conf.EnableGPU
            data.train = gpuArray(data.train);
            data.label = gpuArray(data.label);
        end
        [l, g] = loss_with_gradient_single(data, net, conf);
        loss = loss + l;      
        grad = grad + g;
    end
    loss = gather(loss / num_files);
    grad = gather(grad / num_files);
else
    error('Invalid number of outputs.');
end

end


