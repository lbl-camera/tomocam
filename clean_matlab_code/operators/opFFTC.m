function op = opFFTC(m,n)
% OPFFTC  One-dimensional fast Fourier transform (FFT).
%
%    OPFFTC(M,N) create a one-dimensional centered normalized Fourier transform
%    operator for Matrices of length N.
% SM LBL 09

op = @(x,mode) opFFT_intrnl(m,n,x,mode);


function y = opFFT_intrnl(m,n,x,mode)
checkDimensions(m*n,m*n,x,mode);
if mode == 0
   y = {m*n,m*n,[1,1,1,1],{'FFT'}};
elseif mode == 1
   y=fftshift(fft(fftshift(reshape(x,m,n),1)),1)/sqrt(size(x,1));
   y=y(:);
else
   y=ifftshift(ifft(ifftshift(reshape(x,m,n),1)),1)*sqrt(size(x,1));
   y=y(:);
%   y = ifft(x) * sqrt(length(x));
end
