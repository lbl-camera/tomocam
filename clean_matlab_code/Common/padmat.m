function y=padmat(x,siz,valu)
% function y=padmat(x,size,vals)
% pads x to size with constant values (vals), 
% centers the matrix in the middle.
%
%
n=siz(1);
if length(siz)<2
    m=n;
elseif length(siz)==2
    m=siz(2);
else
    [n m]=size(siz);
end
% [n m]

[N,M]=size(x);
% if nargin<2;
%     m=n;
% end
if nargin<3
    valu=0;
end
% 
%  if n<N n=N;end
%  if m<M m=M;end

y=zeros(n,m)+valu;

y(1:N,1:M)=x;
% size(y)
%y=circshift(y,[fix(n/2)-fix(N/2),fix(m/2)-fix(M/2)]);
y=circshift(y,[fix((n-N)/2),fix((m-M)/2)]);

 