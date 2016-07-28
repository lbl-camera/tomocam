function [A,preprocessop]=forwarmodel_v2(qq,tt,center,delta_xy,delta_r)
% function [A,preprocessop]=forwarmodel(qq,tt)

% Kernel radius
k_r=3;beta =2*pi*2;  %kernel size 2*kr+1

[Ns,nangles]=size(qq);

%[gnuqradon,gnuqiradon,P,opGNUFFT,opprefilter]=gnufft_init_op(Ns,qq,tt,beta,k_r,0);
%[~,~,P,opGNUFFT,opprefilter]=gnufft_init_op_v2(Ns,qq,tt,beta,k_r,center,ones(size(qq)),delta_r,delta_xy,Ns);
%P.opprefilter = opprefilter;

[~,~,P,opGNUFFT]=gnufft_init_spmv_op_v2(Ns,qq,tt,beta,k_r,center,ones(size(qq)),delta_r,delta_xy,Ns);

opFPolyfilter = opFPolyfit(nangles,Ns,P.opprefilter);
A.M=opFoG(opGNUFFT);
A.M=opFoG(opFPolyfilter,opGNUFFT);

A.signalSize=[Ns Ns];
A = completeOps(A);

% forward model: from image to pre-filtered data
preprocessop.image2radon=@(x) P.gnuradon(x);
preprocessop.radon2image=@(x) P.gnuiradon(x);
