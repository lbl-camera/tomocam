clear;

gpuDevice([]);
gpuDevice(2);

addpath operators
addpath gpu
addpath gnufft
addpath Common

Ns=2560;
nangles=180;

signal = padmat(generateAngiogram(Ns/2,Ns/2),[Ns,Ns]);

 Dt=round(180/nangles); %spacing in degrees
[tt,qq]=meshgrid(0:Dt:180-Dt,(1:(Ns))-floor((Ns+1)/2)-1);

% Kernel radius
k_r=2;beta =3*pi*1.0;
[gnuqradon,gnuqiradon,P,opGNUFFT]=gnufft_init_spmv_op(Ns,qq,tt,beta,k_r);
opFPolyfilter = opFPolyfit(nangles,Ns,P.opprefilter);

Fmsk=ones(Ns,nangles);
Fmsk(Ns/2+randi(round(Ns/4),5)-round(Ns/8),:)=0;

data.signal = signal;

data.M=opFoG(opGNUFFT);
data.M=opFoG(opFPolyfilter,opGNUFFT);

% note that real data is 
real_data=P.gnuradon(reshape(data.signal,[Ns,Ns]));
data.b=P.opprefilter(real_data(:),2);


data = completeOps(data);
TV = opDifference(data.signalSize);

  
%%  
% Set solver parameters
opts.maxIts           = 20;
opts.maxLSIts         = 150;
opts.gradTol          = 1e-30;

opts.weightTV         = 0.001;
opts.weightLp         = 0.01;
opts.pNorm            = 1;
opts.qNorm            = 1;
opts.alpha            = 0.01;
opts.beta             = 0.6;
opts.mu               = 1e-12;

x0=data.reconstruct(data.M(data.b,2));

x=x0(:);
msk1=padmat(ones(Ns*3/4),[1 1]*Ns);
x=x.*msk1(:);
subplot(1,2,1);
cropimg=@(img) img(Ns/4+(1:Ns/2),Ns/4+(1:Ns/2));
imagesc(cropimg((abs(x0)+.1).^.5)); axis image
tic;
for i=1:1
    x = solveTV(data.M, data.B, TV, data.b, x, opts);
    y = data.reconstruct(x);
    tm=toc/i;
    subplot(1,2,2);
    imagesc((abs(cropimg(y))+.1).^.5);axis image
    title(sprintf('Iteration %d, timeperiter=%g',i*opts.maxIts,tm));
    drawnow;
    pause
end
%%
 ttime=toc;
 fprintf('total time=%g\n',ttime);