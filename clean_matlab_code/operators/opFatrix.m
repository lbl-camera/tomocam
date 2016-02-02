function op = opFatrix(A)
% OPMATRIX  Apply arbitrary matrix as operator.
%
%    OPMATRIX(A,OPINFO) creates an operator that performs
%    matrix-vector multiplication with matrix A. Optional parameter
%    OPINFO can be used to override the default operator
%    information returned when querying the operator, or provide a 
%    string giving the interpretation of the matrix. When OPINFO is
%    a cell array the first entry is the operator name followed by
%    additional information for that operator type (this option is
%    mostly provided for internal use by other operators). 

%   Copyright 2008, Ewout van den Berg and Michael P. Friedlander
%   http://www.cs.ubc.ca/labs/scl/sparco
%   $Id: opMatrix.m 1040 2008-06-26 20:29:02Z ewout78 $

opinfo = {'Fatrix', []};

op = @(x,mode) opMatrix_intrnl(A,opinfo,x,mode);


function y = opMatrix_intrnl(Gn,opinfo,x,mode)
% m = size(A,1);
% n = size(A,2);

m=Gn.M;
n=prod(Gn.Nd);

checkDimensions(m,n,x,mode);
if mode == 0
%     c =~isreal(Gn);
c=true;
   y = {m,n,[c,1,c,1],opinfo};
elseif mode == 1
%    y = A * x;
%     y=nufft(x(:),Gn);
     y=nufft(reshape(x,Gn.Nd),Gn);
     y=y(:);
else
%       y = (x' * A)';
      y=nufft_adj(x(:),Gn);
     y=y(:);
end
