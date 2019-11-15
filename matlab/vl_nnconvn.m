function varargout = vl_nnconvn(X, F, B, varargin)
%VL_NNCONVN CNN convolution in N spatial dimensions.
%
%   TODO: Update docs for arbitrary number of spatial dimensions
%         instead of [H, W].
%
%   Y = VL_NNCONV(X, F, B) computes the convolution of the image X
%   with the filter bank F and biases B. If B is the empty matrix,
%   then no biases are added. If F is the empty matrix, then the
%   function does not filter the image, but still adds the biases and
%   applies downsampling and padding as explained below.
%
%   X is an array of dimension H x W x C x N where (H,W) are the
%   height and width of the image stack, C is the number of feature
%   channels, and N is the number of images in the batch.
%
%   F is an array of dimension FW x FH x FC x K where (FH,FW) are the
%   filter height and width and K the number o filters in the bank. FC
%   is the number of feature channels in each filter and must match
%   the number of feature channels C in X. Alternatively, FC can
%   *divide* the C; in this case, filters are assumed to form G=C/FC
%   *groups* of equal size (where G must divide K). Each group of
%   filters works on a consecutive subset of feature channels of the
%   input array X.
%
%   [DX, DF, DB] = VL_NNCONV(X, F, B, DY) computes the derivatives of
%   the operator projected onto P. DX, DF, DB, and DY have the same
%   dimensions as X, F, B, and Y, respectively. In particular, if B is
%   the empty matrix, then DB is also empty.
%
%   VL_NNCONV() implements a special *fully-connected* mode: when the
%   support of the filters matches exactly the support of the input
%   image, the code uses an optimized path for faster computation.
%
%   VL_NNCONV(..., 'option', value, ...) accepts the following
%   options:
%
%   `SpatialDims`:: 2
%     Number of spatial dimensions.
%
%   `Stride`:: 1
%     Set the output stride or downsampling factor. If the value is a
%     scalar, then the same stride is applied to both vertical and
%     horizontal directions; otherwise, passing [STRIDEY STRIDEX]
%     allows specifying different downsampling factors for each
%     direction.
%
%   `Pad`:: 0
%     Set the amount of input padding. Input images are padded with zeros
%     by this number of pixels before the convolution is
%     computed. Passing [TOP BOTTOM LEFT RIGHT] allows specifying
%     different padding amounts for the top, bottom, left, and right
%     sides respectively. Passing a single scalar applies the same
%     padding to all borders.
%
%   `Dilate`:: 1
%     Set the kernel dilation factor. Passing [DILATEY DILATEX] allows
%     specifying different dilation factors for Y and X. Filters are
%     dilated by inserting DILATE-1 zeros between filter elements. For
%     example, the filter
%
%       [1 3]
%       [2 4]
%
%     is implicitly treated as
%
%       [1 0 3]
%       [0 0 0]
%       [2 0 4]
%
%     by setting DILATE equal to 2.
%
%   The filter size must be not larger than the padded image, i.e.
%
%     1 <= FH <= H + PADTOP + PADBOTTOM,
%     1 <= FW <= W + PADLEFT + PADRIGHT.
%
%   The output a is an array of dimension YH x YW x K x N of N images
%   with K feature challens and size:
%
%     YH = floor((H + (PADTOP+PADBOTTOM) - FH)/STRIDEY) + 1,
%     YW = floor((W + (PADLEFT+PADRIGHT) - FW)/STRIDEX) + 1.
%
%   Accounting for dilation, the formulas become:
%
%     YH = floor((H + (PADTOP+PADBOTTOM) - FH*(DILATEY-1) -1)/STRIDEY) + 1,
%     YW = floor((W + (PADLEFT+PADRIGHT) - FW*(DILATEX-1) -1)/STRIDEX) + 1.
%
%   Arguments can be SINGLE or DOUBLE and CPU or GPU arrays; however,
%   they must all be of the same type (unless empty).
%
%   ## CUDNN SUPPORT
%
%   If compiled in, the function will use cuDNN convolution routines
%   (with the exception of asymmetric left-right or top-bottom
%   padding that are not supported by cuDNN). You can use the 'NoCudnn' option
%   to disable cuDNN or 'Cudnn' to activate it back again (the choice
%   sticks until MATLAB purges the MEX files for any reason).
%
%   Some cuDNN algorithms may use a very large amount of memory on the
%   GPU (workspace). By default, MatConvNet limits this to 512MB.  To
%   change this behavior, use the `CudnnWorskpaceLimit` option to
%   specify the maximum size of the workspace in bytes. Set this
%   parameter +inf to remove the limit and use the `Verbose` flag to
%   check how much memory is being used.

% Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg.
% Copyright (C) 2015, 2016 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% opts.SpatialDims = 2;
% opts.Stride = 1;
% opts.Pad = 0;
% opts.Dilate = 1;

% Find out mode.
% Y = VL_NNCONV(X, F, B, ...)
% [DX, DF, DB] = VL_NNCONV(X, F, B, DY, ...)
forward = (nargout == 1);

opts.SpatialDims = 2;
[opts, conv_args] = vl_argparse(opts, varargin(1+~forward:numel(varargin)));

% if opts.SpatialDims == 2
%     conv_args = [varargin(1:1-forward) conv_args];
%     [varargout{1:nargout}] = vl_nnconv(X, F, B, conv_args{:});
%     return;
% end

conv_opts.Stride = 1;
conv_opts.Pad = 0;
conv_opts.Dilate = 1;
conv_opts = vl_argparse(conv_opts, conv_args);

xsz = spatial_size(X);
fsz = spatial_size(F);
C = num_channels(X);
N = num_images(X);
FC = num_channels(F);
K = num_images(F);

% TODO: Preproc stride, pad, dilate.
stride = conv_opts.Stride;
pad = conv_opts.Pad;
dilate = conv_opts.Dilate;
% TODO: Remove assertions once implemented.
assert(stride == 1);
assert(dilate == 1);
% assert(pad == 0);  % Zero padding corresponds to 'valid' convolution.
if isscalar(pad)
    pad = repmat(pad, [1 2*numel(xsz)]);
elseif numel(pad) == opts.SpatialDims
    pad = [pad; pad];
    pad = pad(:)';
end
assert(numel(pad) == 2*opts.SpatialDims);

if forward
    % Flip filter to implement correlation via convolution.
    F = flip_spatial_dims(F);
%   The output a is an array of dimension YH x YW x K x N of N images
%   with K feature challens and size:
%     YH = floor((H + (PADTOP+PADBOTTOM) - FH)/STRIDEY) + 1,
%     YW = floor((W + (PADLEFT+PADRIGHT) - FW)/STRIDEX) + 1.
%     YH = floor((H + (PADTOP+PADBOTTOM) - FH*(DILATEY-1) -1)/STRIDEY) + 1,
%     YW = floor((W + (PADLEFT+PADRIGHT) - FW*(DILATEX-1) -1)/STRIDEX) + 1.
    % TODO: Check input types and sizes.
    X = padarray(X, pad(1:2:end), 0, 'pre');
    X = padarray(X, pad(2:2:end), 0, 'post');
    ysz = floor((xsz + pad(1:2:end) + pad(2:2:end) - fsz)./stride) + 1;
    Y = zeros([ysz K N]);
    % Fill Y.
    for i_x = 1:N
        X_i = get_image(X, i_x);
        for i_f = 1:K
            F_i = get_image(F, i_f);
%             B_i = get_image(B, i_f);
            B_i = B(i_f);
            Y_i = convn(X_i, F_i, 'valid') + B_i;
            Y = set_image(Y, i_f, i_x, Y_i);
        end
    end
    varargout{1} = Y;
else
    DY = varargin{1};
    DX = zeros(size(X), 'like', X);
    DF = zeros(size(F), 'like', F);
    DB = ones(size(B), 'like', B) .* sum(DY(:))/K;
    
    X = padarray(X, pad(1:2:end), 0, 'pre');
    X = padarray(X, pad(2:2:end), 0, 'post');
    
    ysz = floor((xsz + pad(1:2:end) + pad(2:2:end) - fsz)./stride) + 1;
    Y = zeros([ysz K N]);
    % Fill DF.
    F_i = ones([spatial_size C], 'like', F);
    for i_x = 1:N
        X_i = get_image(X, i_x);
        Y_i = convn(F_i, X_i, 'valid') .* DY;
%         sum()
%         Y_i 
            Y = set_image(Y, i_f, i_x, Y_i);
%         end
    end
    varargout{1} = Y;
    
    varargout{1} = DX;
    varargout{2} = DF;
    varargout{3} = DB;
end

    function sz = spatial_size(x)
        sz = arrayfun(@(d) size(x, d), 1:opts.SpatialDims);
    end
    function n = num_channels(x)
        n = size(x, opts.SpatialDims + 1);
    end
    function n = num_images(x)
        n = size(x, opts.SpatialDims + 2);
    end
    function x_i = get_image(x, i)
        x_subs.type = '()';
        x_subs.subs = repmat({':'}, 1, opts.SpatialDims + 2);
        x_subs.subs{end} = i;
        x_i = subsref(x, x_subs);
    end
    function x = set_image(x, i_f, i_x, x_i)
        x_subs.type = '()';
        x_subs.subs = repmat({':'}, 1, opts.SpatialDims + 2);
        x_subs.subs{end-1} = i_f;
        x_subs.subs{end} = i_x;
        x = subsasgn(x, x_subs, x_i);
    end
    function x = flip_spatial_dims(x)
        for i = 1:opts.SpatialDims + 1
            x = flip(x, i);
        end
    end
end
