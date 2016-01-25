load('ExampleData/CommandlineData.mat');
 % Type of rendering
%    options.RenderType = 'shaded';
    options.RenderType = 'mip';
    % color and alpha table
    options.AlphaTable=[0 0 0 0 0 1 1 1 1 1];
    options.ColorTable=[1 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0]; 
    V1=V(37+(1:74),37+(1:74));
    
    % Viewer Matrix
%    options.Mview=makeViewMatrix([0 0 0],[0.25 0.25 0.25],[0 0 0]);
%%
for ii=1:180;
    
    options.Mview     =viewmtx(ii*2,45*sin(ii/180*pi*4),30);
I=render(V,options)';
    imshow(I)
    drawnow
end

    %%