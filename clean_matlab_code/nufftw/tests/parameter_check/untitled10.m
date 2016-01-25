load ../data/rt_spiral_03.mat
% This data file contains the non-uniform sample locations k, the k-space
% data d and the density compensation factors w
k = [real(k(:)) imag(k(:))];
d = d(:);
w = w(:);

N = 128; % output image size
maxerr = 1e-3; % maximum aliasing error allowed

imagedirect = read_image('~/Documents/nuFFTW/out/imagedirect.dd');



alpha = 1.2:0.2:2; % grid oversampling ratio
W = 3.5:0.5:7; % kernel width (empirically found to match alpha and maxerr, according to Beatty's paper)
beta_fract = 0.9:0.02:1.06;

e = zeros(N,N,length(alpha),length(W),length(beta_fract));
for ialpha = 1:length(alpha)
    G = ceil(alpha(ialpha)*N); % oversampled grid size
    nstart = floor((G-N)/2); % nstart is the index within the oversampled image where the FOV starts.
    for iW = 1:length(W)
        for ibeta = 1:length(beta_fract)
            beta = pi*sqrt(W(iW)^2/alpha(ialpha)^2*(alpha(ialpha)-0.5)^2-0.8)*beta_fract(ibeta); % Kaiser-Bessel kernel shape parameter, best suited for maxerr, according to Beatty's paper
            gridded_data = nuFFT_grid_2D(k, d.*w, N, alpha(ialpha), W(iW), beta, maxerr);
            image_nodeap = ifftshift(ifft2(gridded_data));
            image_nodeap = image_nodeap(nstart+(1:N),nstart+(1:N));

            gridded_impulse = nuFFT_grid_2D([0 0], 1, N, alpha(ialpha), W(iW), beta, maxerr);
            deap = ifftshift(ifft2(gridded_impulse));
            deap = deap(nstart+(1:N),nstart+(1:N));

            image_deap = image_nodeap ./ deap;
            e(:,:,ialpha,iW,ibeta) = get_image_errors(imagedirect, image_deap);
        end
    end
end
      
save edata e alpha W beta_fract