function [xx,yy,rr]=make_grid(img,ctr,shift)
% function [xx,yy,rr]=make_grid(img,ctr,shift)
% makes a grid of xx,yy centered around ctr
% ctr = [x,y] changes the center of xx,yy,rr
% ctr = 1     centers around the middle
% ctr = 2     centers around the middle-1 
% shift = 1   does an fftshift
%
% SM, LBL 09

[M,N]=size(img);

if M*N==1            %create M x N matrix
    M=img;N=img;
elseif M*N==2
    M=img(2);N=img(1);
end

[xx,yy]=meshgrid(1:M,1:N);

if nargin>1
    if numel(ctr)>1 %2D coordinates
        xx=xx-ctr(1);
        yy=yy-ctr(2);
    elseif ctr==1
        xx=xx-ceil(N/2);
        yy=yy-ceil(M/2);
    elseif ctr==2
        xx=xx-ceil(N/2)-1;
        yy=yy-ceil(M/2)-1;
        
    end
%     if ctr==.5
% %    xx=floor(xx);
% ^    yy=floor(yy);    
end
if nargin==3
    if shift
    xx=fftshift(xx);
    yy=fftshift(yy);
    end
end

    
if nargout>2
    rr=sqrt(xx.^2+yy.^2);
end

  