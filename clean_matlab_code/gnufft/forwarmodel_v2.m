function [A,preprocessop]=forwarmodel_v2(qq,tt,center,delta_xy,delta_r)
% function [A,preprocessop]=forwarmodel(qq,tt)

% Kernel radius
k_r=3;beta =2*pi*2;  %kernel size 2*kr+1

[Ns,nangles]=size(qq);

%[gnuqradon,gnuqiradon,P,opGNUFFT]=gnufft_init_spmv_op_v2(Ns,qq,tt,beta,k_r,center,delta_xy,delta_r);
[gnuqradon,gnuqiradon,P,opGNUFFT]=gnufft_init_spmv_op_v2(Ns,qq,tt,beta,k_r,center,ones(size(projective2d)),delta_r,delta_xy,Ns)

opFPolyfilter = opFPolyfit(nangles,Ns,P.opprefilter);
% 
%data.signal = signal;

A.M=opFoG(opGNUFFT);
A.M=opFoG(opFPolyfilter,opGNUFFT);

A.signalSize=[Ns Ns];
A = completeOps(A);

% forward model: from image to pre-filtered data
preprocessop.image2radon=@(x) P.gnuradon(x);%reshape(x,[Ns,Ns]));
preprocessop.radon2q=@(x) P.opprefilter(x(:),2);
preprocessop.shearlet2image=@(x) A.reconstruct(A.M(x,2));
preprocessop.radon2image=@(x) P.gnuiradon(x);
