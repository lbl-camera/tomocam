function vec = complex2vec(cplx)

vec = zeros(numel(cplx)*2,1);
vec(1:2:end) = real(cplx(:));
vec(2:2:end) = imag(cplx(:));