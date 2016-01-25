function op = opPfith(A,B,opinfo)
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

if nargin < 3
  opinfo = {'Pfith', []};
elseif ischar(opinfo)
  opinfo = {'Pfith', opinfo};
end

op = @(x,mode) opPfith_intrnl(A,B,opinfo,x,mode);


function y = opPfith_intrnl(A,B,opinfo,x,mode)
[m n]= size(A);
% % n = size(A,2);
%   size(x)
%   m 
%   n
[nx nt]=size(B);

checkDimensions(n*floor(length(x)/n),m*floor(length(x)/m),x,mode);
if mode == 0
    c =~isreal(A);
    y = {nx*nt,nx*nt,[c,1,c,1],opinfo};
elseif mode == 1
    y = reshape(x,length(x)/m,m)*A;
    %    y=y';size(y)
    y=y(:);
else
    y = (A* reshape(x,length(x)/m,m)' )';
    y=y(:);
end
