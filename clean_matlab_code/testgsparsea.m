% SpMV using cusp with gpuArrays
%
% Stefano Marchesini, Lawrence Berkeley National Laboratory, 2013


% generate matrix
M=15000;
a=rand(M)>.97;
a((1:M)+(0:M-1)*M)=true; %keep diagonal
% symmetrize
a=a'+a;

% view matrix
spy(a);axis([0 300 0 300])
fprintf('initialize sparse indices...\n');
%%
% get CSC format
[col,row]=find(a);
 
 
ncol=numel(col);
% generate random entries
vals=rand(ncol,1)+1i*rand(ncol,1);

% construct matrix from coo for comparison, 
bs=sparse(row,col,vals);
 
/* get input arguments */
 A.num_rows
 A.num_cols
 A.indx
 A.col
 A.val
 A.format='coo';
 
 B=gcsr(A);
 

% we swap rows and columns because cusp uses CSR


ptr=[0;(find(diff(row)));numel(row)]; % pointer for CSR
nptr=numel(ptr);

% generate random vector
x=(rand(nptr-1,1)+1i*rand(nptr-1,1));

%%
% push to gpu
fprintf('push to gpu...\n');
gptr=gpuArray(int32(ptr));
gcol=gpuArray(int32(col-1));
gval=gpuArray(single(vals));
gx=gpuArray(single(x));
%%
% 
%%
% timing
ntries=100;
tfac=1e3/ntries; %millisec
fprintf('gpu complex...');
gy=gspmv(gval,gcol,gptr,gx); % one first try...
tic;
for ii=1:ntries;
gy=gspmv(gval,gcol,gptr,gx);
end
tgpu=toc*tfac; %microsec
fprintf('%g msec\n',tgpu);
fprintf('cpu complex...');
tic;
for ii=1:ntries;
y=bs*x;
end
tcpu=toc*tfac;
fprintf('%g msec\n',tcpu);
fprintf('speedup=%g\n',tcpu/tgpu)

yy=gather(gy);
reldif=norm(yy-y,'fro')/norm(y,'fro');
fprintf('relative precision error=%g\n',reldif)

title(sprintf('zoom of %d x %d,nnz=%g, \ntimes cpu=%g$\\mu s$ \n gpu=%g , difference:%g$\\mu s$, \n ratio=%g\n',...
    M,M,ncol, tcpu,tgpu,tcpu/tgpu,reldif),'FontSize',20,'Interpreter','latex')
drawnow;

%title(sprintf('tcpu(mu sec)=%g, tgpu($\mu$ sec)=%g , ratio=%g\n',tcpu*1e4,tgpu*1e4,tcpu/tgpu))
%return

% % real case
 bsr=sparse(row,col,real(vals));
 xr=real(x);
 gvalr=gpuArray(single(real(vals)));
 gx=gpuArray(single(xr));
 gyr=gspmv(gvalr,gcol,gptr,gx);
 fprintf('gpu real...');
tic;
for ii=1:ntries;
gyr=gspmv(gvalr,gcol,gptr,gx);
end;
tgpur=toc*tfac; 
fprintf('%g msec\n',tgpur);
 fprintf('cpu real...');
tic;
for ii=1:ntries;
yr=bsr*xr;
end
tcpur=toc*tfac; %microsec
fprintf('%g msec\n',tcpur);
str=sprintf('tcpu(r,c)=(%g,%g)$\\mu s$ \n tgpu(r,c)=(%g,%g) $\\mu s$, \n ratio(r,c)=(%g,%g)\n',tcpur,tcpu,tgpu,tgpur,tcpur/tgpur,tcpu/tgpu);
fprintf('speedup (r,c) (%g,%g)\n',tcpur/tgpur,tcpu/tgpu);

title(sprintf(['zoom of %d  %d,nnz=%g, \n ' ...
    'tcpu(r,c)=(%g,%g)$\\mu s$ \n tgpu(r,c)=(%g,%g) $\\mu s$, \n ratio(r,c)=(%g,%g)'],...
    M,M,ncol, tcpur,tcpu,tgpu,tgpur,tcpur/tgpur,tcpu/tgpu),'FontSize',20,'Interpreter','latex')

return


