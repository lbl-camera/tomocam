function op = opPolyfit(nangles,nscans,Transform,opinfo)
% opPolyfit(nangles,nscans, Transform, OPINFO) 
%
%  it returns operator that computes 
%    data_out=Polynomial_Filter*data_in;
%
% the polynomial filter operator in radon space along pixel dimension
% removes constant offsets for each pixel plus slow time variation of the
% pixel intensity along the angle.
% 
% it builds  a vandermonde matrix
% vander2=@(x)[ x.^2 x.^1 x.^0]';
% V=vander2(linspace(0,1,nangles)'/nangles);
% and makes polynomial filter: 
% Filter= I-V'*(V*V')^(-1)*V;
%
% it returns operator that computes 
%    data_out=Filter*data_in;
% prefilter F may be used to transform from fourier slices
%    data_out=Transform' *  ( Filter* (Transform*data_in ) )
%  
%   opinfo: name of the operator
%   nangles,nscans: size of the radon space
% 
% S. Marchesini, LBNL 2010 
% 


if nargin<4
    opinfo={ 'FPolyFit'};
  if nargin<3
    Transform=[];
  end
end

col=@(x) x(:);

% build vandermonde matrix, assume only quadratic offset
%vander2=@(x)[ x.^2  ]';
%vander2=@(x)[x.^2 x.^1 x.^0]';
%V=vander2(single(gpuArray.linspace(0,1,nangles)'/nangles));
vander2=@(x)[((x)+eps).^0]';

V=real(vander2(single(gpuArray.linspace(0,1,nangles)'/nangles))');

% make polynomial filter: I-V'*(V*V')^(-1)*V;
%Pfilt2=gpuArray.eye(nangles)-V'*(V*V')^(-1)*V;
Pfilt2=gpuArray.eye(nangles)-V'*((V'*V)\V);
%Pfilt2=gpuArray.eye(nangles);

if nargin < 4
  opinfo = {'Polyfit', []};
elseif ischar(opinfo)
  opinfo = {'Polyfit', opinfo};
end

op = @(x,mode) opPolyfit_intrnl(x,mode);


function y = opPolyfit_intrnl(x,mode)
  checkDimensions(nangles*nscans,nangles*nscans,x,mode);
if mode == 0
    y = {nangles*nscans,nangles*nscans,[1,1,1,1],opinfo};
else
    if isempty(Transform)
    y = reshape(x,length(x)/nangles,nangles)*Pfilt2;
    y=y(:);
    else
%        jk=reshape(Transform(x,1),length(x)/nangles,nangles);
        
%        jk1=reshape(Transform(x,1),length(x)/nangles,nangles);
        
    y = Transform(col(reshape(Transform(x,1),length(x)/nangles,nangles)*Pfilt2),2);
%    y = col(reshape(x,length(x)/nangles,nangles)*Pfilt2);
    end
end

end
end
