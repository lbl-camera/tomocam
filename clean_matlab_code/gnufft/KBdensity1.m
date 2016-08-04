function DD=KBdensity1(qq,tt,KB,nj,Ns)


%divide in blocks
nb=100; %TODO : Why 100 ? Venkat 

[nt,nq]=size(qq);
% % crop outer points:

%crop repeated angles
ii=(tt(:,1))-min(tt(:,1))<180;
qq1=qq(ii,:);
tt1=tt(ii,:);
qq1=-fliplr(qq1);


[xi,yi]=pol2cart(tt1*pi/180,qq1);

%now assume all radial things have the same environment:
% note that if nd is even, this is not true for the outer circle(s),  
ind_in=find(tt1==90 &qq1>=0);

%and points within +/- nj to the y-axis:
ind_out=find(abs(xi)<nj+1 & qq1>-nj-1);


% mult factor: # block/#pixels:
mmx=(nb-1)/(max(yi(ind_in))+2*nj+2);

i2b=@(x) floor((x+2*nj+1)*mmx)+2; %add 2 block to the left

indx=i2b(yi(ind_in));%floor((xi(ind_in))*mmx)+2;
S0=sparse(indx,ind_in,ones(size(indx)),nb+4,numel(xi));
 
%now find the 'outer' points
%these include innter points:
S1=S0;

% shift by -nj
indx=i2b(yi(ind_out)-nj);
S1=S1+sparse(indx,ind_out,ones(size(indx)),nb+4,numel(xi));

% shift by +nj
indx=i2b(yi(ind_out)+nj);
S1=S1+sparse(indx,ind_out,ones(size(indx)),nb+4,numel(xi));

indx=i2b(yi(ind_out)-floor(nj/2));
S1=S1+sparse(indx,ind_out,ones(size(indx)),nb+4,numel(xi));
indx=i2b(yi(ind_out)+floor(nj/2));
S1=S1+sparse(indx,ind_out,ones(size(indx)),nb+4,numel(xi));

%crop outer blocks
% S0=S0(2:end-1,:);
% S1=S1(2:end-1,:);

%%
%finally, compute density
D=zeros(nnz(S0),1);
indD=0;

tic
for ii=1:nb+2;
    [~,jj_in] =find(S0(ii,:));
    [~,jj_out]=find(S1(ii,:));
    
    diffx=bsxfun(@plus,xi(jj_in),-xi(jj_out)');
    diffy=bsxfun(@plus,yi(jj_in),-yi(jj_out)');
    
    D(indD+(1:length(jj_in)))=sum(KB(diffx).*KB(diffy),1);
    indD=indD+length(jj_in);
end
% normalization of the kernel...
D=D/sum(sum(KB(-nj:nj)'*KB(-nj:nj)));

%ccc=sum(KB(-nj:nj))

%shift the index to the center of a Ns by Ns array
% xi = xi+floor((Ns+1)/2)+1; 
% yi = yi+floor((Ns+1)/2)+1;
nsm=floor((Ns-1)/2);
% cropm=(xi>=-nsm & yi>=-nsm & xi<=Ns-nsm-1 & yi<=Ns-nsm-1);

% replicate matrix, 
% the output 

if mod(nq,2)
    DD=D([end-1:-1:2, 1:end],ones(nt,1))';
    'hi'
%     DD=DD(cropm);
else
    DD=D([end:-1:2, 1:end-1],ones(nt,1))';
%     DD=DD(cropm);
end


return

