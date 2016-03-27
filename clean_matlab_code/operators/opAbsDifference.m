function op = opAbsDifference(size)
%OPABSDIFFERENCE  Generation of absolute value of difference function
%
%   OPABSDIFFERENCE(SIZE) returns the handle of a difference function
%   that takes the arguments X and MODE. When MODE is 1 the vector
%   X is reshaped to SIZE and the difference is taken in the ROW
%   and COLUMN direction and output in an matrix with two
%   columns. When SIZE is a scalar, only the difference along the
%   COLUMNS is taken and a single column vector is returned.
%   When MODE is 2 the reverse operation is done, except that the
%   offset is lost.

%   Copyright 2008, Ewout van den Berg and Michael P. Friedlander
%   http://www.cs.ubc.ca/labs/scl/sparco
%   $Id: opDifference.m 1040 2008-06-26 20:29:02Z ewout78 $

if length(size) == 1
   op = @(x,mode) opDifference_intrnl1(size(1),x,mode);
elseif length(size) == 2
   op = @(x,mode) opDifference_intrnl2(size(1),size(2),x,mode);
else
  error('Higher dimensional difference operator not supported yet');
end


function y = opDifference_intrnl1(m,x,mode)
if (mode == 1)
   y       = x([2:m,m]) + x;
else
   y       =  x([1,1:m-1]) + x;
   y(1)    = x(1);
   y(m)    =  x(m-1);
end

function y = opDifference_intrnl2(m,n,x,mode)
if (mode == 1)
   z       = reshape(x,m,n);
   zx      = z([2:m,m],:) + z;
   zy      = z(:,[2:n,n]) + z;
   y       = [zx(:), zy(:)];
else
   xr      = reshape(x(:,1),m,n);
   zx      =  xr([1,1:m-1],:) + xr;
   zx(1,:) = xr(1,:);
   zx(m,:) =  xr(m-1,:);
   
   xr      = reshape(x(:,2),m,n);
   zy      =  xr(:,[1,1:n-1]) + xr;
   zy(:,1) =  xr(:,1);
   zy(:,n) =  xr(:,n-1);
   
   y       = reshape(zx + zy, m*n, 1);
end

%% Code by venkat - March 2016. Modifying to have a 8 point neighborhood
%% TODO : Boundary conditions for diagonal neighbors

% function y = opDifference_intrnl2(m,n,x,mode)
% 
% %weights associated with each voxel pair
% wnorm = 4+2*sqrt(2);
% w1 = 1/wnorm;
% w2 = sqrt(2)/wnorm;
% 
% if (mode == 1)
%    z       = reshape(x,m,n);
%    zx      = z([2:m,m],:) - z;
%    zy      = z(:,[2:n,n]) - z;
%    zxy1    = [z(2:m,2:n) z(1:m-1,n);z(m,:)] - z;
%    zxy2    = [z(1:end-1,1) z(2:m,1:n-1);z(end,:)] - z;
%    y       = [w1*zx(:), w1*zy(:), w2*zxy1(:), w2*zxy2(:)];
% else
%    xr      = reshape(x(:,1),m,n);
%    zx      =  xr([1,1:m-1],:) - xr;
%    zx(1,:) = -xr(1,:);
%    zx(m,:) =  xr(m-1,:);
%    
%    xr      = reshape(x(:,2),m,n);
%    zy      =  xr(:,[1,1:n-1]) - xr;
%    zy(:,1) = -xr(:,1);
%    zy(:,n) =  xr(:,n-1);
%    
%    xr        = reshape(x(:,3),m,n);
%    zxy1      =  [xr(1,:); xr(2:m,1) xr(1:m-1,1:n-1)] - xr; %xr([1,1:m-1],[1,1:n-1]) - xr;
%    zxy1(1,:) = -xr(1,:);
%    zxy1(m,:) =  xr(m-1,:);
%    zxy1(:,1) = -xr(:,1);
%    zxy1(:,n) = xr(:,n-1);
%    
%    xr      = reshape(x(:,4),m,n);
%    zxy2      = [xr(1,:);xr(1:m-1,2:n) xr(2:m,n)] - xr;
%    zxy2(:,1) = -xr(:,1);
%    zxy2(:,n) =  xr(:,n-1);
%    zxy2(1,:) = -xr(1,:);
%    zxy2(m,:) =  xr(m-1,:);
%       
%    y       = reshape((w1)*zx + (w1)*zy + (w2)*zxy1 + (w2)*zxy2, m*n, 1);
% 
% end