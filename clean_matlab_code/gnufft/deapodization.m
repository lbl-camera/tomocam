function dpz=deapodization(Ns,KB)
% function dpz=deapodization(Ns,KB)
% returns deapodization factor for kernel KB
%

xx=(1:(Ns))-Ns/2-1;%(-NsO/2:NsO/2-1);
%dpz=circshift(ifftshift(ifft2(fftshift(KB(xx,0)'*KB(xx,0)))),[10 0]);
dpz=circshift(ifftshift(ifft2(fftshift(KB(xx,0)'*KB(xx,0)))),[0 0]);
% assume oversampling, do not divide outside box in real space:

%msk=logical(padmat(ones(floor([1 1]*Ns*2/3)),[Ns Ns])); %---mask
%TO DO : Why these numbers like 2/3 ? Venkat Jan 2016
msk = logical(CreateCircularBinaryMask(Ns,Ns,Ns/2,Ns/2,Ns/2));

% ii=find(~msk); clear msk


dpz=single(dpz);
dpz(~msk)=1;            %keep the value outside box
dpz=1./dpz;            %deapodization factor truncated

dpz=dpz/dpz(Ns/2+1,Ns/2+1); %----scaling
%dpz=dpz./max(abs(dpz(:)));
