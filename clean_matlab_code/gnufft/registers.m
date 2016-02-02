% s_per_b(nbins) int32
% b_points_x(npoints) single 
% b_points_y (npoints) single
%  b_loc(nbins) int32 block coordinate (round(b_loc/Ns),mod(b_loc,Ns))
% b_offset(nbins) location on the indexed array
% b_dim_x(nbins) int32
% s_in_bin(npoints);  indexed array of points

% grid size
Ns=300;
grid=[Ns Ns];
% kernel radius and width
k_r=2; kw=2*kr+1; kk=(-k_r:k_r);
[kblut,KB,KB1,KB2D]=KBlut(k_r,2*pi*1,256);


% number of points
ns=128;
val=rand(1,ns);
qq=1:ns;
tt=zeros(1,ns);
[x,y]=pol2cart(tt,qq);
xi=round(x);dx=x-xi;
yi=round(y);dy=y-yi;

psi0=zeros(Ns,Ns);psi0(Ns/2,20+(1:ns))=val;
filter=KB2D(kk',kk);
psi=conv2(psi0,KB2D(kk',kk),'same');

nvec=ns;

aa=zeros(kw,kw,nvec);

for ii=ns:-1:1;
aa(:,:,ii)=KB2D(kk'+dx(ii),kk+dy(ii)).*val(ii);
end


% linear sum
bb=aa;
bb(:,2:end,1:end-1)=bb(:,2:end,1:end-1)+bb(:,1:end-1,2:end); %sum every 2 layers
bb(:,3:end,1:end-2)=bb(:,3:end,1:end-2)+bb(:,1:end-2,3:end); % sum the top 4 layers
bb(:,1,5:end)=bb(:,1,5:end)+bb(:,5,1:end-4); % sum the 5th layer
% now put them together
%qq=([bb(:,1:4,1) squeeze(bb(:,1,5:end)) bb(:,5 ,end-3) bb(:,5,end-2) bb(:,5,end-1) bb(:,5,end)]);
qq=([bb(:,1:4,1) squeeze(bb(:,1,5:end)) squeeze( bb(:,5 ,end-3:end)) ]);

psi2=zeros(Ns,Ns);psi2(Ns/2-3+(1:5),20-2+(1:132))=qq;
psi2=zeros(Ns,Ns);psi2(Ns/2-3+(1:5),20-2+(1:132))=qq;

% diagonal sum
bb=aa;
bb(2:end,2:end,1:end-1)=bb(2:end,2:end,1:end-1)+bb(1:end-1,1:end-1,2:end); %sum every 2 layers
bb(3:end,3:end,1:end-2)=bb(3:end,3:end,1:end-2)+bb(1:end-2,1:end-2,3:end); % sum the top 4 layers
bb(1,1,5:end)=bb(1,1,5:end)+bb(5,5,(5:end)-4); % sum the 5th layer
bb(1,2,5:end)=bb(1,2,5:end)+bb(5,4,(5:end)-3); % sum the 5th layer
bb(2,1,5:end)=bb(2,1,5:end)+bb(4,5,(5:end)-3); % sum the 5th layer
bb(1,3,5:end)=bb(1,3,5:end)+bb(5,3,(5:end)-2); % sum the 5th layer
bb(3,1,5:end)=bb(3,1,5:end)+bb(3,5,(5:end)-2); % sum the 5th layer
bb(4,1,5:end)=bb(4,1,5:end)+bb(2,5,(5:end)-1); % sum the 5th layer
bb(1,4,5:end)=bb(1,4,5:end)+bb(5,2,(5:end)-1); % sum the 5th layer




bb(1,2:4,5:end)=bb(1,2:4,5:end)+bb(5,3:5,(5:end)-3); % sum the 5th layer
bb(2:4,1,5:end)=bb(2:4,1,5:end)+bb(3:5,5,(5:end)-3); % sum the 5th la



% now put them together
%qq=([bb(:,1:4,1) squeeze(bb(:,1,5:end)) bb(:,5 ,end-3) bb(:,5,end-2) bb(:,5,end-1) bb(:,5,end)]);
qq=([bb(:,1:4,1) squeeze(bb(:,1,5:end)) squeeze( bb(:,5 ,end-3:end)) ]);

psi2=zeros(Ns,Ns);psi2(Ns/2-3+(1:5),20-2+(1:132))=qq;

imagesc([bb(1:4,1:4,1) [0 ;bb(3:end,3,3)] [0;0;bb(4:end,3,4)] [0;0;0;bb(5,5,5)]])






b_points_x=

bdim=int32(8);
nbins=16;
npts=5;
ksize=2; %kernel



ptx=rand(5)*(bdim+ksize*2)-ksize;
pty=rand(5)*(bdim+ksize*2)-ksize;

ptval=rand(1)+1i*rand(1);

a=0;
for ii=1:npts;a=a+KB2D((0:bdim)'+round(-pty(ii)),(0:bdim)+round(-ptx(ii)));end



a=(KB2D((0:bdim)'-pty,(0:bdim)-ptx))


grid = [Ns,Ns];

KB(0,-3:.1:3);
