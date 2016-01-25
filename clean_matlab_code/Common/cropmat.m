function y=cropmat(x,siz)
% function y=cropmat(x,size)
% crops x to size around center
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

[N,M]=size(x);

y=zeros(n,m);

y=x(round(N/2-n/2)+(1:n),round(M/2-m/2)+(1:m));

% size(y)
%y=circshift(y,[fix(n/2)-fix(N/2),fix(m/2)-fix(M/2)]);
%y=circshift(y,[fix((n-N)/2),fix((m-M)/2)]);

