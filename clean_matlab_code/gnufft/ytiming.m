% s_per_b(nbins) int32
% b_points_x(npoints) single 
% b_points_y (npoints) single
%  b_loc(nbins) int32 block coordinate (round(b_loc/Ns),mod(b_loc,Ns))
% b_offset(nbins) location on the indexed array
% b_dim_x(nbins) int32
% s_in_bin(npoints);  indexed array of points


bdim=int32(8);
nbins=16;
npts=5;
ksize=2; %kernel 
kw=ksize*2+1;
% generate random points include points in halo
ptx=rand(npts)*(bdim+ksize*2)-ksize;
pty=rand(npts)*(bdim+ksize*2)-ksize;
ptval=rand(npts)+1i*rand(npts);

a=0;
for ii=1:npts;a=a+ptval(ii)*KB2D((0:bdim)'+round(-pty(ii)),(0:bdim)+round(-ptx(ii)));end


a=(KB2D((0:bdim)'-pty,(0:bdim)-ptx))


grid = [Ns,Ns];

KB(0,-3:.1:3);
