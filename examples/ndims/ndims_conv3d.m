function res = ndims_conv3d()
%NDIMS_CONV3 Test 3D convolution

% Simple
net.layers{1} = struct(...
    'name', 'conv1', 'type', 'conv', ...
    'weights', {{randn(3,3,3,2,'single'), randn(2,1,'single')}}, ...
    'pad', 0, 'stride', 1, 'dilate', 1);

net.layers{2} = struct(...
    'name', 'relu1', 'type', 'relu');

net = vl_simplenn_tidy(net);

data = randn(8, 8, 8, 1, 'single');
res = vl_simplenn(net, data);

end
