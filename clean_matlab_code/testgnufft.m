% Load source image
N=2048*3/2;
N=512;

% G = padmat(interp2(single(imread('cameraman.tif')),N/256),4*n);
  G = padmat(phantom(floor(N*4/3)),N*2);
[Ns,Mq]=size(G);

% Irregular grid in polar coordinates
%Dt=.25; %spacing in degrees
Dt=3; %spacing in degrees
[tt,qq]=meshgrid(0:Dt:180-Dt,(1:(Ns))-floor((Ns+1)/2)-1);
nangles=size(tt,2);

% Kernel radius
beta =3*pi*1.0;
k_r=2;
[gnuradon,gnuiradon,qtXqxy,qxyXqt]=gnufft_init(Ns,qq,tt,beta,k_r);


Gxy=gpuArray(complex(single(G)));

subplot(1,2,1);
imagesc(real((gnuradon(Gxy))))
axis square
subplot(1,2,2);
imagesc(real(gnuiradon(gnuradon(Gxy))))
% ax=caxis;caxis([.1 1]*ax(2)); %hide noise!
%caxis([600 5000])
axis square
colormap hot
%%
ntries=100;
tic;
for ii=1:ntries;
    jk=gnuradon(Gxy);
end
t1=toc/ntries*1e3; %msec

for ii=1:ntries;
    jk2=gnuiradon(jk);
end
t2=toc/ntries*1e3; %msec
fprintf(sprintf('size(%g,%g),angles=%g, \ntime=(radon:%g, iradon:%g) (msec)',Ns,Ns,nangles,t1,t2));
%
%return
%%
tic;C=radon(G,tt(1,:)); tradoncpu=toc*1e3;
%imagesc(abs(cropmat(C,size(qq))));
tic;C1=complex((single(cropmat(C,size(qq)))));
D=iradon(C1,tt(1,:));tiradoncpu=toc*1e3;
%%   
fprintf(sprintf('time (msec) (radon:%0.0g, iradon:%g) (msec)\n',tradoncpu,tiradoncpu));
title(sprintf('time (cpu(msec),gpu(msec),speedup)  \nradon:(%0.0f, %0.0f, %0.0f)  \niradon:(%0.0f, %0.0f, %0.0f) ',tradoncpu,t1,tradoncpu/t1,tiradoncpu,t2,tiradoncpu/t2));