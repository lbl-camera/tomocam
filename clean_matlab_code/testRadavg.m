M=1500;
% % an image that  we want to average
% a=rand(M);
%generate an index based on the radius;
ii=(1:M)-M/2;
rr=sqrt(bsxfun(@plus,ii.^2,ii'.^2)); %radius
nnz=numel(rr);

nr=.1;
rri=round(rr*nr)/nr;
[ru,u2uns,r2u]=unique(rri); % we need unique identifier: 
%%
P=op_accuamarray(r2u);
Pnorm=P*gpuArray.ones([P.nnz 1],'single');



%%

for ii=1:1e5;
    ga=gpuArray.rand(M,'single');
    ga_avg=(P*ga)./Pnorm;
    ga2=reshape(ga_avg(r2u),[1 1]*M);
subplot(1,2,1);
    imagesc(abs(ga));axis image off
subplot(1,2,2);
    imagesc(abs(ga2).^3);axis image off
    drawnow;

end

%%
