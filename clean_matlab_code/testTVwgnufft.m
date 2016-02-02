figon=false;

%if true
% Load source image
%N=2048*3/2;
N=1024;
nangles=30;

% G = padmat(interp2(single(imread('cameraman.tif')),N/256),4*n);
%  G = padmat,phantom(floor(N*4/3)),N*2);
%  G = padmat(phantom('Modified Shepp-Logan',floor(N*4/3)),N*2);
 G = padmat(generateAngiogram(floor(N*4/3),floor(N*4/3)),N*2);
  G=G/norm(G(:));
  [Ns,Mq]=size(G);

% Irregular grid in polar coordinates
%Dt=.25; %spacing in degrees
%Dt=9; %spacing in degrees
Dt=round(180/nangles); %spacing in degrees
[tt,qq]=meshgrid(0:Dt:180-Dt,(1:(Ns))-floor((Ns+1)/2)-1);
%nangles=size(tt,2);

% Kernel radius
beta =3*pi*1.0;
k_r=2;
%[gnuradon,gnuiradon,qtXqxy,qxyXqt,opGNUFFT]=gnufft_init(Ns,qq,tt,beta,k_r);
 [gnuqradon,gnuqiradon,P,opGNUFFT]=gnufft_init_op(Ns,qq,tt,beta,k_r);

%[gnuradon_spmv,gnuiradon_spmv,qtXqxy_spmv,qxyXqt_spmv]=gnufft_init_spmv(Ns,qq,tt,beta,k_r);


Gxy=gpuArray(complex(single(G)));
%return
%%
% now add ring noise
%Polynomial fit along each vertical stripe
%vander4=@(x) [x.^4 x.^3 x.^2 x.^1 x.^0]';
%A=vander4(linspace(-1,1,nangles)');

if false
vander2=@(x) [ x.^2 x.^1 x.^0]';
A=vander2(single(gpuArray.linspace(-1,1,nangles)'/nangles*2));
ringfactors=gpuArray(single([randn(Ns,1)*1e1 randn(Ns,1)*1e4, randn(Ns,1)*1e-1+1]));

ringnoise=ringfactors*A;


Gradon=gnuradon(Gxy)+ringnoise;
end
    
% Gradon=gnuqradon(Gxy);
% Gradon=gnuqradon(Gxy);
 Gradon=P.gnuradon(Gxy);
col=@(x) x(:);

Gqradon= P.datafilt(Gradon);

% polynomial filter
%Pfilt2=gpuArray.eye(size(A,2))-A'*(A*A')^(-1)*A;

%data.signal=gpuArray.randn(Ns,Ns);
data.signal=(Gxy);

%data.op.polyfit=opPolyfit(nangles,Ns);
%data.sizeM=[1 1]*Ns*nangles;
data.op.gnufft=opGNUFFT;
%data.sizeA=[1 1]*Ns*Ns;
%data.B=@(x,mode) opDirac_intrnl(Ns*Ns,x,mode);
%data.sizeB=[1 1]*Ns*Ns;
%data.M=opFoG(data.op.polyfit,data.op.gnufft);
%data.M=opFoG(data.op.polyfit,data.op.gnufft);
%data.M=opFoG(data.op.gnufft);
data.M=data.op.gnufft;
data.b=data.M(data.signal(:),1);

%data.signalSize=[Ns, Ns];
% data.reconstruct=@(x) reshape(x,data.signalSize);

 data = completeOps(data);
  
TV = opDifference(data.signalSize);
  

  % Set solver parameters
  opts.maxIts           = 10;
  opts.maxLSIts         = 150;
  opts.gradTol          = 1e-30;

  opts.weightTV         = 1e0;
  opts.weightLp         = 0.000;
  opts.pNorm            = 1;
  opts.qNorm            = 1;
  opts.alpha            = 0.6;
  opts.beta             = 0.6;
  opts.mu               = 1e-3;

%  
  % Give instructions to the user.
%   fprintf('This script calls "solveTV" five times,\n');
%   fprintf('each with a maximum of 10 iterations.\n');
%   fprintf('Ignore the messages "ERROR EXIT"\n');
%  input('Press "Return" to continue.');
    %x = randn(prod(data.signalSize),1);
%    x0=data.reconstruct(data.M(data.b,1));
    x=data.reconstruct(data.M(data.b,2));
%    x0=x;
    x0=Gxy;
    x=x0(:);
    subplot(1,2,1);
    imagesc(abs(x0)); colormap hot;
    axis image

    'hi'
  for i=1:20
    x = solveTV(data.M, data.B, TV, data.b, x, opts);
    y = data.reconstruct(x);
    subplot(1,2,2);
    imagesc(abs(y)); axis image
    title(sprintf('Iteration %d',i));
    drawnow;
  end
  return
  %%
  opts = spgSetParms('optTol',1e-4);
  tau = 0;   % Initial one-norm of solution
  sigma = 0; % Go for a basis pursuit solution
 z = spgl1(P.A, P.b, tau, sigma, [], opts);
 