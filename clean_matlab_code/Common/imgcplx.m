function [ h,img ] = imgcplx( X ,alfa,p0)
% function [ h,img ] = imgcplx( X ,alfa,p0)
% 
% function to plot complex variables
% input: X: thing to plot
%        alpha: brightness contrast
%        p0: phase origin


% take the phase
if nargin==1
    alfa=1; p0=0;
elseif nargin==2
    p0=0;
end

ang=angle(X.*exp(1i*p0))/pi/2+1/2;

% take amplitude
amp=abs(X)./max(abs(X(:)));
% change "contrast"
amp=amp.^alfa;

% convert from Hue Sat Val to RGB
img=hsv2rgb(cat(3,ang,ang*0+1,amp));
h=imagesc(img);

end

