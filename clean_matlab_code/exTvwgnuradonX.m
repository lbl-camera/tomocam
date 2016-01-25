addpath operators
addpath gpu
addpath gnufft
addpath Common

%data=generateProblem(502);
 Ns=1024; nangles=90;

  signal = padmat(generateAngiogram(Ns/2,Ns/2),[Ns,Ns]);
%  signal = padmat(phantom('Modified Shepp-Logan',floor(N*4/3)),Ns);
% signal = padmat(phantom('Modified Shepp-Logan',floor(Ns/2)),Ns);
% signal = generateAngiogram(n,n);
 % Generate the data
% pdf    = genPDF([Ns,Ns],5,0.33,2,0.1,0);
% mask   = genSampling(pdf,10,60);
% mask= logical(full(fftshift(RadialLines(Ns,nangles))));

 Dt=round(180/nangles); %spacing in degrees
[tt,qq]=meshgrid(0:Dt:180-Dt,(1:(Ns))-floor((Ns+1)/2)-1);

% Kernel radius
k_r=2;beta =3*pi*1.0;
[gnuqradon,gnuqiradon,P,opGNUFFT]=gnufft_init_spmv_op(Ns,qq,tt,beta,k_r);
opFPolyfilter = opFPolyfit(nangles,Ns,P.opprefilter);
%opFPolyfilter = opFPolyfit(nangles,Ns,P.qtXrt);

Fmsk=ones(Ns,nangles);
Fmsk(Ns/2+randi(round(Ns/4),5)-round(Ns/8),:)=0;

%return
%opFPolyfilter = opFmsk(Fmsk);
 %%
% signal = generateAngiogram(n,n);
%
 %end
%%
% Set up the problem
%data.pdf             = pdf;
%data.mask            = mask;
data.signal          = signal;
%data.op.mask         = opMask(mask);
%data.op.padding      = opPadding([n,n],[n,n]);
%data.op.fft2d        = opFFT2C(Ns,Ns);
% data.M               = opFoG(data.op.mask, data.op.padding, ...
%                              data.op.fft2d);

%data.M               = opFoG(data.op.mask,  data.op.fft2d);
%data.M=opFoG(opGNUFFT);
data.M=opFoG(opFPolyfilter,opGNUFFT);


%data.b               = data.M(reshape(data.signal,[Ns*Ns,1]),1);
% note that real data is 
real_data=P.gnuradon(reshape(data.signal,[Ns,Ns]));
data.b=P.opprefilter(real_data(:),2);
data.b=opFPolyfilter(data.b,1);

%data.b               = data.M(reshape(data.signal,[Ns*Ns,1]),1);
%data.b=P.datafilt(reshape(real_data,[Ns,nangles]));
%data.b=P.datafilt(real_data);
%data.b=data.b(:);

data                 = completeOps(data);
%  op = opFPolyfit(nangles,nscans,F)
 TV = opDifference(data.signalSize);

  % Set solver parameters
  opts.maxIts           = 100;
  opts.maxLSIts         = 150;
  opts.gradTol          = 1e-30;

  opts.weightTV         = 0.001;
  opts.weightLp         = 0.01;
  opts.pNorm            = 1;
  opts.qNorm            = 1;
  opts.alpha            = 0.01;
  opts.beta             = 0.6;
  opts.mu               = 1e-12;
  % Solve
  x0=data.reconstruct(data.M(data.b,2));
  
%  x = randn(prod(data.signalSize),1);
  x=x0(:);
  msk1=padmat(ones(Ns/2),[1 1]*Ns);
  x=x.*msk1(:);
%figure;colormap hot
subplot(1,2,1);
cropimg=@(img) img(Ns/4+(1:Ns/2),Ns/4+(1:Ns/2));
    imagesc(cropimg((abs(x0)+.1).^.5)); axis image 
%caxis([0 .5])
    tic;
  for i=1:1
    x = solveTV(data.M, data.B, TV, data.b, x, opts);
    y = data.reconstruct(x);
    tm=toc/i;
subplot(1,2,2);  
    imagesc((abs(cropimg(y))+.1).^.5);axis image 
%    caxis([0 .5])
    title(sprintf('Iteration %d, timeperiter=%g',i*opts.maxIts,tm));
    drawnow;
    pause
  end
%%
 ttime=toc;
 fprintf('total time=%g\n',ttime);