%% mex me
setenv('MW_NVCC_PATH','/usr/local/cuda/bin/nvcc')

fprintf('try also nvcc -I/usr/local/matlab/extern/include -I/usr/local/matlab/toolbox/distcomp/gpu/extern/include   -arch sm_35 -c  gspmv_coo.cu')


fnames=dir('gs*.cu');
% fnames(11)=[];
% fnames(10)=[];
% fnames(7)=[];


for ii=1:numel(fnames)
fprintf('\n----------------------------------------------\n')    
fprintf('\n||             %s      ||\n',fnames(ii).name)    
fprintf('\n----------------------------------------------\n')    
mex(fnames(ii).name)
end

%