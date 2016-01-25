function op = opPolyfit(nangles,nscans,opinfo)
% opPolyfit  Apply arbitrary matrix as operator.
%
%    opPolyfit(nangles,OPINFO) creates an operator that performs
%    matrix-vector multiplication with matrix A. Optional parameter
%    OPINFO can be used to override the default operator
%    information returned when querying the operator, or provide a 
%    string giving the interpretation of the matrix. When OPINFO is
%    a cell array the first entry is the operator name followed by
%    additional information for that operator type (this option is
%    mostly provided for internal use by other operators). 

vander2=@(x) [ x.^2 x.^1 x.^0]';
A=vander2(single(gpuArray.linspace(-1,1,nangles)'/nangles*2));
Pfilt2=gpuArray.eye(size(A,2))-A'*(A*A')^(-1)*A;

if nargin < 3
  opinfo = {'Polyfit', []};
elseif ischar(opinfo)
  opinfo = {'Polyfit', opinfo};
end

op = @(x,mode) opPolyfit_intrnl(x,mode);


function y = opPolyfit_intrnl(x,mode)
% nscans=floor(numel(x)/nangles);
  checkDimensions(nangles*nscans,nangles*nscans,x,mode);
if mode == 0
    y = {nangles*nscans,nangles*nscans,[1,1,1,1],opinfo};
else % it is a projector=transpose
    y = reshape(x,length(x)/nangles,nangles)*Pfilt2;
    y=y(:);
end

end
end
