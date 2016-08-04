function dpz=deapodization_v2(Ns,KB1D,Nr_orig)
% function dpz=deapodization(Ns,KB)
% returns deapodization factor for kernel KB with a mask of size Nr_orig
%

xx=(1:(Ns))-Ns/2-1;%(-NsO/2:NsO/2-1);

dpz=ifftshift(ifft2(fftshift(KB1D(xx)'*KB1D(xx))));


dpz=single(dpz);
%dpz(~msk)=0;            %keep the value outside box
dpz=1./dpz;            %deapodization factor truncated

%dpz=dpz/dpz(Ns/2+1,Ns/2+1); %----scaling
%dpz=dpz./max(abs(dpz(:)));
