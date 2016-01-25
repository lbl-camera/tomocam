function [recon]=MBIRTV(projection,init)
%Function to call the TV MBIR code 
%Input : projection : A 2-D array with num_angles X n_det entries
%containing the projection data
%       angles : a list of angles used (1 X num_angles array)
%
g=gpuDevice(2);
reset(g);
%gpuDevice(2);

addpath operators
addpath gpu
addpath gnufft
addpath Common

%Parameters
k_r=2;beta =3*pi*1.0;
% Set solver parameters
opts.maxIts           = 100;
opts.maxLSIts         = 150;
opts.gradTol          = 1e-30;

opts.weightTV         = 0.005*1000;
opts.weightLp         = 0.0;
opts.pNorm            = 1;
opts.qNorm            = 1;
opts.alpha            = 0.01;
opts.beta             = 0.6;
opts.mu               = 1e-12;

%Code starts here
[nangle,Ns]=size(projection');
Dt=180/nangle;

[tt,qq]=meshgrid(0:Dt:180-Dt,(1:(Ns))-floor((Ns+1)/2)-1);
[gnuqradon,gnuqiradon,P,opGNUFFT]=gnufft_init_spmv_op(Ns,qq,tt,beta,k_r);
opFPolyfilter = opFPolyfit(nangle,Ns,P.opprefilter);

data.signal = gpuArray(init);
data.M=opFoG(opGNUFFT);
data.M=opFoG(opFPolyfilter,opGNUFFT);
real_data = gpuArray(projection);
data.b=P.opprefilter(real_data(:),2);
data=completeOps(data);
TV = opDifference(data.signalSize);
x0=data.reconstruct(data.M(data.b,2));
x=x0(:);
msk1=padmat(ones(Ns/2),[1 1]*Ns);
x=x.*msk1(:);
x = solveTV(data.M, data.B, TV, data.b, x, opts);
recon = data.reconstruct(x);