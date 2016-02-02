function nu2 = nu2NEMAphantom(Nx,Ny,Nz,Activity)
%-------------------------------------------------------------------------------
%   nu2NEMAPhantom: create the 3 dimensional NU2 - 2000 NEMA phantom.  
%           Reference: "NEMA Standards Publication NU2 2000: Performance 
%                       Measurements of  Positron Emission Tomographs"
%   
%   The phantom created here is described on a space of 35cm*35cm*18 cm.  
%   Consequently the voxel size is (35/Nx)cm * (35/Ny)cm * (18/Nz)cm
%   The body wall thickness is 3 mm, the lung is modeled as 25 cm radius
%   cylinder in the center of the phantom extending throughput its length
%   Either the emission of transmission phantom can be created. The default
%   transmission phantom is generated if the activity is not specified.
%
%   The emission phantom has 6 spheres of 10, 13, 17, 22, 28 and 37 mm
%   diameters. NEMA recommends the first four cylinders are hot spheres
%   and the last two cold spheres.
%
%   Syntax:     nu2 = nu2NEMAphantom(Nx,Ny,Nz,Activity)
%
%   Inputs:     Nx, Ny, Nz  -   Number of pixels in each dimension
%               Activity    -   A seven element vector.
%   [bkg, sphere10, sphere13, sphere17, sphere22, sphere28, sphere37]
%                                   
%   Output:     nu2         -   3 dimensional NU2 - 2000 NEMA phantom. 
%                               Units: mu 1/mm for transmission images
%                                      input activity units
%
%   Written by Ravi Manjeshwar
%   History:    Oct 22, 2001   -  Transmission phantom
%               Oct 23, 2001   -  Added emission phantom
%               Oct 24, 2001   -  Fixed bug that altered dimensions
%
%-------------------------------------------------------------------------------

if (nargin<3)
    disp('Creating default transmission phantom of dimensions 350*350*180');
    Nx = 350; Ny=350; Nz = 180;
    EmOrTrFlag = 0;
elseif (nargin==4) EmOrTrFlag = 1; 
else EmOrTrFlag = 0;
end

dx = 350/Nx;
dy = 350/Ny;
dz = 180/Nz;

x = [-Nx/2:Nx/2-1]*dx;
x = x(ones(Ny,1),:);
y = [-Ny/2:Ny/2-1]'*dy;
y = y(:,ones(1,Nx));
z = [-Nz/2:Nz/2-1]'*dz;
z = reshape(z,1,1,Nz);

nu2BodyWall = zeros(Nx,Ny);
nu2BodyWall(Ny/2+1-round(35/dy):size(x,1),:) = (sqrt(x(Ny/2+1-round(35/dy):size(x,1),:).^2 + ...
                                         (y(Ny/2+1-round(35/dy):size(x,1),:)+35).^2) < 150);
nu2BodyWall(1:Ny/2-round(35/dy),1:Nx/2+1-round(70/dx)) = ...
    (sqrt( (x(1:Ny/2-round(35/dy),1:Nx/2+1-round(70/dx))+70).^2 + ...
          (y(1:Ny/2-round(35/dy),1:Nx/2+1-round(70/dx))+35).^2 ) < 80);
nu2BodyWall(1:Ny/2-round(35/dy),Nx/2+1+round(70/dx):size(x,2)) = ...
    (sqrt( (x(1:Ny/2-round(35/dy),Nx/2+1+round(70/dx):size(x,2))-70).^2 + ...
          (y(1:Ny/2-round(35/dy),Nx/2+1+round(70/dx):size(x,2))+35).^2 ) < 80);
nu2BodyWall(Ny/2+1-round(115/dy):Ny/2+1,Nx/2-round(70/dx):Nx/2+round(70/dx)) = 1;

nu2SoftTissue(Ny/2+1-round(35/dy):size(x,1),:) = (sqrt(x(Ny/2+1-round(35/dy):size(x,1),:).^2 + ...
                                            (y(Ny/2+1-round(35/dy):size(x,1),:)+35).^2) < 147);
nu2SoftTissue(1:Ny/2-round(35/dy),1:Nx/2+1-round(70/dx)) = ...
        (sqrt( (x(1:Ny/2-round(35/dy),1:Nx/2+1-round(70/dx))+70).^2 + ...
          (y(1:Ny/2-round(35/dy),1:Nx/2+1-round(70/dx))+35).^2 ) < 77);
nu2SoftTissue(1:Ny/2-round(35/dy),Nx/2+1+round(70/dx):size(x,2)) = ...
        (sqrt( (x(1:Ny/2-round(35/dy),Nx/2+1+round(70/dx):size(x,2))-70).^2 + ...
          (y(1:Ny/2-round(35/dy),Nx/2+1+round(70/dx):size(x,2))+35).^2 ) < 77);
nu2SoftTissue(Ny/2+1-round(112/dy):Ny/2+1,Nx/2-round(70/dx):Nx/2+round(70/dx)) = 1;
	
if ( EmOrTrFlag == 0)   % Build transmission NU2-2001 NEMA phantom
%-----------------------------------------------------------------
    % Mass attenuation coefficients for 511 KeV, Units: cm^2/g
	muRhoPMMA       = 9.410e-2;         % PMMA, 
	muRhoSoftTissue = 9.598e-2;         % soft tissue
	muRhoAir        = 8.712e-2;         % air
    muRhoLung       = 9.607e-2;         % lung
    % Density, Units: g/cm^3
    rhoPMMA         = 1.190e+0;
    rhoSoftTissue   = 1.060e+0;
    rhoAir          = 1.205e-3;
    rhoLung         = 1.050e+0;
    
	nu2Lung = find(sqrt(x.^2+y.^2)<25);
	
	nu2BodyWall = (muRhoPMMA*rhoPMMA)*(nu2BodyWall-nu2SoftTissue);
	nu2SoftTissue = (muRhoSoftTissue*rhoSoftTissue)*nu2SoftTissue;
	nu2 = nu2BodyWall+nu2SoftTissue;
	nu2(nu2Lung) = muRhoLung*rhoLung;
	
	nu2 = repmat(nu2/10,[1,1,Nz]);
    
else    % Build emission NU2-2001 NEMA phantom
%-------------------------------------------------

    x = x(:,:,ones(1,1,Nz));
    y = y(:,:,ones(1,1,Nz));
    z = z(ones(Ny,1),ones(1,Nx),:);
    r = 57.2;
    nu2 = Activity(1)*repmat(nu2SoftTissue,[1,1,Nz]);
    ang10 = 60*pi/180;
    sphere10 = find( sqrt((x-r*cos(ang10)).^2 + (y-r*sin(ang10)).^2 + z.^2) <= 10/2 );
    ang13 = 120*pi/180;
    sphere13 = find( sqrt((x-r*cos(ang13)).^2 + (y-r*sin(ang13)).^2 + z.^2) <= 13/2 );
    sphere17 = find( sqrt((x-r).^2 + y.^2 + z.^2) <= 17/2 );
    ang22 = 240*pi/180;
    sphere22 = find( sqrt((x-r*cos(ang22)).^2 + (y-r*sin(ang22)).^2 + z.^2) <= 22/2 );
    ang28 = 300*pi/180;
    sphere28 = find( sqrt((x-r*cos(ang28)).^2 + (y-r*sin(ang28)).^2 + z.^2) <= 28/2 );
    sphere37 = find( sqrt((x+r).^2 + y.^2 + z.^2) <= 37/2 );
    nu2(sphere10) = Activity(2);
    nu2(sphere13) = Activity(3);
    nu2(sphere17) = Activity(4);
    nu2(sphere22) = Activity(5);
    nu2(sphere28) = Activity(6);
    nu2(sphere37) = Activity(7);

end
