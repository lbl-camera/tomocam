function [St,Sn]=radavgop(img)
%  [S,Sn]=radavgop(img);
%  given an image return sparse matrix to do radial avrage
%
%  usage: img_avg=Sn*img(:);                   for 1D plots
%         img_avg2d=reshape(S'*img_avg,Mx,My); for 2D plots;
%



[Mx,My]=size(img);
[xx,yy]=meshgrid((1:Mx)-ceil(Mx/2)-1,(1:My)-ceil(Mx/2)-1); %matrix to compute center of mass
rr=sqrt(xx.^2+yy.^2); %radius

bin=1;
rb=round(rr/bin);
[rbs,u2s,s2u]=unique(round(rr/bin));
S=sparse(s2u,1:Mx*My,0*s2u+1);
avgn=sum(S,2); %normalization

Sn=bsxfun(@times,S,1./avgn);
St=S';

% ravg=Sn*rr(:);

%adavg=Sn*ad(:); %radial average
%adavg2D=reshape(S'*adavg,Mx,My); % copy back onto 2D
 