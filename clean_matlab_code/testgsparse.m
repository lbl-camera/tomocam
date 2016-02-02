% SpMV using cusp with gpuArrays
%
% Stefano Marchesini, Lawrence Berkeley National Laboratory, 2013


% generate matrix
fprintf('initialize sparse indices...\n');
M=4000;
a=complex(rand(M),rand(M));%>.9;
a=a.*(real(a)>.94);
% symmetrize
a=a'+a;
a((1:M)+(0:M-1)*M)=rand(M,1); %keep diagonal

%%


% view matrix
figure(1);
spy(a);
axis([0 1 0 1]*min(300,M));
fprintf('initialize sparse indices...\n');
%
% sparse formats
[col,row,vals]=find(a);
nnz=numel(find(a));

% construct matrix from coo for comparison, 
%%bs=sparse(row,col,vals);
bs=sparse(a);
%%
%double

%%
 A=gcsparse((a),1);
% x=(rand(M,1)+1i*rand(M,1));
 x=rand(M,1);
gx=gpuArray(single(x));

y=bs*x; %cpu SPMV
gy=A*gx; % gpu SPMV
yy=gather(gy);
reldif=norm(yy-y,'fro')/norm(y,'fro');
fprintf('relative precision error=%g\n',reldif)

%
figure(2);
plot(1:M,abs(bs*x),'r',1:M,abs(A*gx),'b.');
legend({'cpu','gpu'});
title('spmv');
title(sprintf('spmv (%d x %d,nnz=%g)*(%d), \n $\\epsilon$:%g',...
    M,M,nnz,M,reldif),'FontSize',20,'Interpreter','latex')

%  return
%%
% % timing
 ntries=100;
 tfac=1e3/ntries; %millisec
 fprintf('gpu complex...');

gy=A*gx; % one first try...

tic;
for ii=1:ntries;
gy=A*gx;
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
figure(1);
spy(a);
axis([0 1 0 1]*min(300,M));

title(sprintf('zoom of %d x %d,nnz=%g, \ntimes cpu=%g$\\mu s$ \n gpu=%g , difference:%g$\\mu s$, \n speedup=%g\n',...
    M,M,nnz, tcpu,tgpu,tcpu/tgpu,reldif),'FontSize',20,'Interpreter','latex')
drawnow;
%%
%return
%title(sprintf('tcpu(mu sec)=%g, tgpu($\mu$ sec)=%g , ratio=%g\n',tcpu*1e4,tgpu*1e4,tcpu/tgpu))
%return

% % real case
 
bsr=real(bs);
 xr=real(x);
 gxr=gpuArray(single(xr));
Ar=real(A);

gyr=Ar*gxr;
 fprintf('gpu real...');
tic;
for ii=1:ntries;
gyr=Ar*gxr;
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

yy=norm(gather(gyr)-yr,'fro')/norm(yr,'fro');
%reldif=norm(yy-y,'fro')/norm(y,'fro');
fprintf('relative precision error=%g\n',reldif)


str=sprintf('tcpu(r,c)=(%g,%g)$\\mu s$ \n tgpu(r,c)=(%g,%g) $\\mu s$, \n ratio(r,c)=(%g,%g)\n',tcpur,tcpu,tgpu,tgpur,tcpur/tgpur,tcpu/tgpu);
fprintf('speedup (r,c) (%g,%g)\n',tcpur/tgpur,tcpu/tgpu);
legend(sprintf(' nnz=%g \nsize(%g,%g)', nnz,M,M));
title(sprintf(['zoom of A, spmv times in $\\mu s$:\n cpu(real,cplx)=(%g,%g) \n gpu(real,cplx)=(%g,%g), \n     speedup(real,complex)=(%g,%g)'],...
   tcpur,tcpu,tgpu,tgpur,tcpur/tgpur,tcpu/tgpu),'FontSize',20,'Interpreter','latex')

return
%%

