addpath gpu
addpath gnufft
addpath operators
addpath  Common

figon=false;


if true
% Load source image
%N=2048*3/2;
N=1024;


if ~exist('nangles','var');nangles=180;end
% nangles=360;
%nangles=720;
nangles=180;
%uniqueness=true;
uniqueness=false;

% G = padmat(interp2(single(imread('cameraman.tif')),N/256),4*n);
%  G = padmat,phantom(floor(N*4/3)),N*2);
  G = padmat(phantom('Modified Shepp-Logan',floor(N*4/3)),N*2);
[Ns,Mq]=size(G);

% Irregular grid in polar coordinates
Dt=180/nangles; %spacing in degrees
[tt,qq]=meshgrid(0:Dt:180-Dt,(1:(Ns))-floor((Ns+1)/2)-1);

 

% Kernel radius
beta =3*pi*1.0;
k_r=2;

tic;
[gnuradon,gnuiradon,qtXqxy,qxyXqt,opGNUFFT]=gnufft_initx(Ns,qq,tt,beta,k_r);
%[gnuradon,gnuiradon,qtXqxy,qxyXqt,opGNUFFT]=gnufft_init(Ns,qq,tt,beta,k_r,uniqueness);
t1=toc;tic;
%[gnuradon_spmv,gnuiradon_spmv,qtXqxy_spmv,qxyXqt_spmv]=gnufft_init_spmv(Ns,qq,tt,beta,k_r);
%t2=toc;

fprintf(sprintf('time=(init:%g)\n',t1));

Gxy=gpuArray(complex(single(G)));
tic;for ii=1:1;Gradon=gnuradon(Gxy);end;toc

tic
GG=gnuiradon(Gradon);
toc


%return
%%
if figon
figure(1)
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
figure(2)
subplot(1,2,1);
%imagesc(real((gnuradon_spmv(Gxy))))
axis square
subplot(1,2,2);
%imagesc(real(gnuiradon_spmv(gnuradon_spmv(Gxy))))
% ax=caxis;caxis([.1 1]*ax(2)); %hide noise!
%caxis([600 5000])
axis square
colormap hot
drawnow
end

end
%%

ntries=100;
tic;
for ii=1:ntries;
    Gradon=gnuradon(Gxy);
end
tradon=toc/ntries*1e3; %msec


tic;
for ii=1:ntries;
    GG=gnuiradon(Gradon);
end
tiradon=toc/ntries*1e3; %msec


fprintf(sprintf('size(%g,%g),angles=%g, \ntime=(radon:%g, iradon:%g) (msec)\n',Ns,Ns,nangles,tradon,tiradon));
%fprintf(sprintf('time=(radon:%g, iradon:%g) (msec) spmv\n',tradon_spmv,tiradon_spmv));
%
return
%%
tic;C=radon(G,tt(1,:)); tradoncpu=toc*1e3;
%imagesc(abs(cropmat(C,size(qq))));
tic;C1=complex((single(cropmat(C,size(qq)))));
D=iradon(C1,tt(1,:));tiradoncpu=toc*1e3;
%%   
fprintf(sprintf('time (msec) (radon:%0.0g, iradon:%g) (msec)\n',tradoncpu,tiradoncpu));
title(sprintf('time (cpu(msec),gpu(msec),speedup)  \nradon:(%0.0f, %0.0f, %0.0f)  \niradon:(%0.0f, %0.0f, %0.0f) ',tradoncpu,tradon,tradoncpu/tradon,tiradoncpu,t2,tiradoncpu/t2));


%%
% now add ring noise
%Polynomial fit along each vertical stripe
%vander4=@(x) [x.^4 x.^3 x.^2 x.^1 x.^0]';
%A=vander4(linspace(-1,1,nangles)');
vander2=@(x) [ x.^2 x.^1 x.^0]';
A=vander2(single(gpuArray.linspace(-1,1,nangles)'/nangles*2));
ringfactors=gpuArray(single([randn(Ns,1)*1e1 randn(Ns,1)*1e4, randn(Ns,1)*1e-1+1]));

ringnoise=ringfactors*A;


Gradon=gnuradon(Gxy)+ringnoise;
col=@(x) x(:);

% polynomial filter
%Pfilt2=gpuArray.eye(size(A,2))-A'*(A*A')^(-1)*A;

%data.signal=gpuArray.randn(Ns,Ns);
data.signal=(Gxy);

data.op.polyfit=opPolyfit(nangles,Ns);
%data.sizeM=[1 1]*Ns*nangles;
data.op.gnufft=opGNUFFT;
%data.sizeA=[1 1]*Ns*Ns;
%data.B=@(x,mode) opDirac_intrnl(Ns*Ns,x,mode);
%data.sizeB=[1 1]*Ns*Ns;
%data.M=opFoG(data.op.polyfit,data.op.gnufft);
data.M=opFoG(data.op.polyfit,data.op.gnufft);
data.b=data.M(data.signal(:),1);



%data.signalSize=[Ns, Ns];
% data.reconstruct=@(x) reshape(x,data.signalSize);

 data = completeOps(data);
  
TV = opDifference(data.signalSize);
  

  % Set solver parameters
  opts.maxIts           = 10;
  opts.maxLSIts         = 150;
  opts.gradTol          = 1e-30;

  opts.weightTV         = 0.001;
  opts.weightLp         = 0.01;
  opts.pNorm            = 1;
  opts.qNorm            = 1;
  opts.alpha            = 0.01;
  opts.beta             = 0.6;
  opts.mu               = 1e-12;

%%  
  % Give instructions to the user.
  fprintf('This script calls "solveTV" five times,\n');
  fprintf('each with a maximum of 10 iterations.\n');
  fprintf('Ignore the messages "ERROR EXIT"\n');
  input('Press "Return" to continue.');
    %x = randn(prod(data.signalSize),1);
%    x0=data.reconstruct(data.M(data.b,1));
    x=data.reconstruct(data.M(data.b,2));
    x=x(:);
  for i=1:5
    x = solveTV(data.M, data.B, TV, data.b, x, opts);
    y = data.reconstruct(x);
    imagesc(abs(y));
    title(sprintf('Iteration %d',i));
    drawnow;
  end
  

  
%return
    ntries=1000;
xx=gpuArray(single(0:Ns-1));     fftshift1D=(-1).^xx';  
grmask=gpuArray(abs(qq)<size(qq,1)/4*3/2);
rtXqt=@(Gqt) ifftshift(bsxfun(@times,fftshift1D,ifft(Gqt)),1).*grmask;
  
GradonQT=rtXqt(Gradon);                     

tic;for ii=1:ntries;    GG=qxyXqt(GradonQT);end;tiradon=toc/ntries*1e3
