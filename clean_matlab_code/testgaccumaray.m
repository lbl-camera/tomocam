% accumarray vs accumarg
%%
ntries=100;
tfac=1e3/ntries;
M=5e6;
vals=rand(M,1)+1i*rand(M,1);
indx=randi(100,M,1);

accumarg=op_accuamarray(indx);
gvals=gpuArray(single(vals));
tic;
for ii=1:ntries;
    aval=accumarray(indx,vals);
end; tca=toc*tfac; 
tic;
for ii=1:ntries;
    gaval=accumarg(gvals);
end; tga=toc*tfac; 

caval=gather(gaval);
reldif=norm(aval-caval,'fro')/norm(aval(:));

fprintf('accumarray, times(cpu,gpu)(%g,%g), speedup: %g, numerical difference=%g \n',tca,tga,tca/tga,reldif);


%%