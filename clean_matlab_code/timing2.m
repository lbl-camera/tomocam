
  
%return
    ntries=1000;
xx=gpuArray(single(0:Ns-1));     fftshift1D=(-1).^xx';  
grmask=gpuArray(abs(qq)<size(qq,1)/4*3/2);
rtXqt=@(Gqt) ifftshift(bsxfun(@times,fftshift1D,ifft(Gqt)),1).*grmask;
  
GradonQT=rtXqt(Gradon);                     

tic;for ii=1:ntries;    GG=qxyXqt(GradonQT);end;tiradon=toc/ntries*1e3
