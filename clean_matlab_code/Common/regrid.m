function b=regrid(a,n,empty)
% function b=regrid(a,n [,empty])
% 
% resample the image n times, 
% if empty=1 places 0 around every pixel, otherwise
% just copies a pixel n times in x and y
%
% SM 2003

if n==0;return;end
[nx,ny]=size(a);

nbx=n(1);
nby=n(end);

%%
h=zeros(nbx,nby);
if nargin>2
    if empty==1
        h(1,1)=1;
    else
        h(:)=1;
    end
else 
    h(:)=1;
end
b=kron(a,h);
% 
% %%
% 
% %copy the pixels
% b=a(:,:,ones(nbx,1),ones(nby,1));
% if nargin>2
%     if empty==1
%         %set them to 0
%         b(:,:,2:end,:)=0;
%         b(:,:,:,2:end)=0;
%     end
% end
% 
% %back to 2D
% b=reshape(permute(b,[4,1,3,2]),nx*nby,ny*nbx);
% %imagesc(b);
