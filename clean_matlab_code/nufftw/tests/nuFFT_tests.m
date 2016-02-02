load ../data/rt_spiral_03.mat
% This data file contains the non-uniform sample locations k, the k-space
% data d and the density compensation factors w

k = [real(k(:)) imag(k(:))];

d = d(:);
w = w(:);

N = 128; % output image size
alpha = 2; % grid oversampling ratio
maxerr = 1e-3; % maximum aliasing error allowed
W = 3.84326171875; % kernel width (empirically found to match alpha and maxerr, according to Beatty's paper)
beta = pi*sqrt(W^2/alpha^2*(alpha-0.5)^2-0.8); % Kaiser-Bessel kernel shape parameter, best suited for maxerr, according to Beatty's paper
G = ceil(alpha*N); % oversampled grid size
nstart = floor((G-N)/2); % nstart is the index within the oversampled image where the FOV starts.

m = compute_gridding_matrix(k, N, alpha, W, beta, maxerr);

imagedirect = read_image('~/Documents/nuFFTW/out/imagedirect.dd');

% adjoint
gridded_data = m*(d.*w);
gridded_data = reshape(gridded_data,G,G);
%[gridded_data,m] = nuFFT_grid_2D(k, d.*w, N, alpha, W, beta, maxerr);
image_nodeap = ifftshift(ifft2(gridded_data));
image_nodeap = image_nodeap(nstart+(1:N),nstart+(1:N));

gridded_impulse = nuFFT_grid_2D([0 0], 1, N, alpha, W, beta, maxerr);
deap = ifftshift(ifft2(gridded_impulse));
deap = deap(nstart+(1:N),nstart+(1:N));

image_deap = image_nodeap ./ deap;
e = get_image_errors(imagedirect,image_deap);

im = image_deap(:);


% forward
im1 = reshape(im,128,128);
im2 = im1./conj(deap)/256^2;
im3 = zeros(G);
im3(nstart+(1:N),nstart+(1:N)) = im2;
im4 = fft2(fftshift(im3));
im5 = im4(:);
d4 = m'*im5;

(d.*w)'*d4-im(:)'*im(:)

x = (-N/2):(N/2-1);
deap3 = sinc(sqrt((pi*W*x/G).^2-beta^2)/pi);
[deapx,deapy] = meshgrid(deap3,deap3);
deap3 = deapx.*deapy;
[x,y] = meshgrid(x,x);
xy = x+y;
x = (-1).^xy;
deap3 = deap3.*x;
image_deap3 = image_nodeap./deap3;
ek = get_image_errors(imagedirect,image_deap3);

imk = image_deap3(:);

% forward
im1 = reshape(imk,128,128);
im2 = im1./conj(deap3)/256^2;
im3 = zeros(G);
im3(nstart+(1:N),nstart+(1:N)) = im2;
im4 = fft2(fftshift(im3));
im5 = im4(:);
d5 = m'*im5;

(d.*w)'*d5-imk(:)'*imk(:)




% kernel bounds, normalized to grid size
W2G = W/2/G;

% presample the interpolation kernel
S = ceil(sqrt(0.37/maxerr)/alpha);

% k-space locations of presampled kernel
k_kb = (-W2G-(1/S/G)):(1/S/G):(W2G+(1/S/G));

% presampled Kaiser-Bessel kernel, formula 4 in Beatty's paper
KB = G/W*besseli(0,beta*sqrt(1-(2*k_kb*G/W).^2));

% find k-space locations that are outside of the kernel width and zero them
% out
a = abs(k_kb) > W2G;
KB(a) = 0;


KB(S*G)=0;
kb = ifftshift(ifft(KB));
kb2 = kb((S*G)/2+((-N/2):(N/2-1))+1);
kb2 = kb2 .* sinc(((-N/2):(N/2-1))/(S*G)).^2;
[kbx,kby] = meshgrid(kb2,kb2);
kb2d = kbx.*kby;
image_deap2 = image_nodeap./kb2d;
x = 1:N;
[x,y] = meshgrid(x,x);
xy = x+y;
x = (-1).^xy;
image_deap2 = image_deap2 .* x;
e = get_image_errors(imagedirect,image_deap2);





M = 10;
k = rand(M,2)-0.5;
d = (rand(M,1)-0.5)+i*(rand(M,1)-0.5);
w = 1;
N = 10;



x = w.*d;
x2 = m*x/normest(m);
y = ifft(x2)*sqrt(size(m,1));
y4 = fft(y)/sqrt(size(m,1));
z = m'*y4/normest(m);
x'*z-y'*y


x = w.*d;
x2 = m*x/normest(m);
x3 = reshape(x2,sqrt(size(m,1)),sqrt(size(m,1)));
y = ifft2(x3)*sqrt(size(m,1));
y2 = y(:);
y3 = reshape(y2,sqrt(size(m,1)),sqrt(size(m,1)));
y4 = fft2(y3)/sqrt(size(m,1));
y5 = y4(:);
z = m'*y5/normest(m);
x'*z-y2'*y2