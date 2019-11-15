
args = {'SpatialDims', 2};
conv_args = {'Stride', 1, 'Pad', 1, 'Dilate', 1}; 

% forward
C = 1;
N = 1;
K = 3;
xsz = [3 4 C N];
fsz = [3 3 C K];
X = rand(xsz);
F = rand(fsz);
% F = ones(fsz);
B = rand([1 K]);
% B = zeros([1 K]);
Y = vl_nnconvn(X, F, B, conv_args{:}, args{:});
Y2 = vl_nnconv(X, F, B, conv_args{:});

% Y - Y2
assert(all(abs(Y(:) - Y2(:)) <= sqrt(eps)));

% backward
% DY = rand(size(Y), 'like', Y);
DY = ones(size(Y), 'like', Y);
[DX, DF, DB] = vl_nnconvn(X, F, B, DY, conv_args{:}, args{:});
[DX2, DF2, DB2] = vl_nnconv(X, F, B, DY, conv_args{:});

DX - DX2
DF - DF2
DB - DB2
