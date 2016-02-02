function Mask=CreateCircularBinaryMask(m,n,yc,xc,r)
% function to create a circular mask for an image
% Inputs: m - number of rows (y axis), n - number of columns (x axis) 
%         yc - center of circle in y (row)
%         xc - center of circle in x (col) 
%         r - radius 
% Outputs: Mask an m X n binary image with 1's in region of selection and 0
% elsewhere
% This code is from: http://www.mathworks.com/matlabcentral/newsreader/view_thread/146031

[xx yy]=meshgrid(1:n,1:m);
Mask=sqrt((xx-xc).^2+(yy-yc).^2)<=r;
