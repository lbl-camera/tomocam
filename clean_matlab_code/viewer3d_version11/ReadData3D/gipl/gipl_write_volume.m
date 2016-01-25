function gipl_write_volume(I,fname,scales)
% Function for writing Guys Image Processing Lab (Gipl) files.
% 
% gipl_write_volume(volume, filename, voxelsize in mm)
%
% examples:
% I=uint8(rand(64,64,64)*256);
%
% 1: gipl_write_volume(I);
% 2: gipl_write_volume(I,'random.gipl',[2 2 2];


image_types=struct('double',65,'float',65,'single',64,'logical',1,'int8',7,'uint8',8,'int16',15,'uint16',16,'int32',32,'uint32',31);
bit_lengths=struct('double',64,'float',64,'single',32,'logical',1,'int8',8,'uint8',8,'int16',16,'uint16',16,'int32',32,'uint32',32);

% Filename
    if((nargin<2)||strcmp(fname,'')), 
        [filename, pathname] = uiputfile('*.gipl', 'Write gipl-file'); 
        fname = [pathname filename]; 
    end
% Sizes
    sizes=size(I);
    while(length(sizes)<4), sizes=[sizes 1]; end
% Scales
    if(exist('scales','var')==0), scales=ones(1,length(sizes)); end;
    while(length(scales)<4), scales=[scales 0]; end
% Offset
    offset=256;
% Image Type
    image_type=getfield(image_types,class(I));
% File size
    fsize=offset+prod(sizes)*getfield(bit_lengths,class(I))/8;
% Patient
    patient='Generated by Matlab';  
    while(length(patient)<80), patient=[patient ' ']; end
% Matrix
    matrix=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
% orientation
    orientation=0;
% voxel_min
    voxmin=min(I(:));
% voxel_max
    voxmax=max(I(:));
% origing
    origin=[0 0 0 0];
% RescaleIntercept
    RescaleIntercept=0;
% RescaleSlope
    RescaleSlope=1;
% interslicegap
    interslicegap=0;
% user_def2
    user_def2=0;
% par2
    par2=0;
% magic_number
    magic_number=4026526128;

trans_type{1}='binary'; trans_type{7}='char'; trans_type{8}='uchar'; trans_type{15}='short';
trans_type{16}='ushort'; trans_type{31}='uint'; trans_type{32}='int'; trans_type{64}='float'; 
trans_type{65}='double'; trans_type{144}='C_short'; trans_type{160}='C_int'; trans_type{192}='C_float'; 
trans_type{193}='C_double'; trans_type{200}='surface'; trans_type{201}='polygon';

trans_orien{0+1}='UNDEFINED'; trans_orien{1+1}='UNDEFINED_PROJECTION'; trans_orien{2+1}='AP_PROJECTION'; 
trans_orien{3+1}='LATERAL_PROJECTION'; trans_orien{4+1}='OBLIQUE_PROJECTION';  trans_orien{8+1}='UNDEFINED_TOMO'; 
trans_orien{9+1}='AXIAL'; trans_orien{10+1}='CORONAL'; trans_orien{11+1}='SAGITTAL'; trans_orien{12+1}='OBLIQUE_TOMO';

disp(['filename : ' num2str(fname)]);   
disp(['filesize : ' num2str(fsize)]);
disp(['sizes : ' num2str(sizes)]);
disp(['scales : ' num2str(scales)]);
disp(['image_type : ' num2str(image_type) ' - ' trans_type{image_type}]);
disp(['patient : ' patient]);
disp(['matrix : ' num2str(matrix)]);
disp(['orientation : ' num2str(orientation) ' - ' trans_orien{orientation+1}]);
disp(['voxel min : ' num2str(voxmin)]);
disp(['voxel max : ' num2str(voxmax)]);
disp(['origing : ' num2str(origin)]);
disp(['RescaleIntercept : ' num2str(RescaleIntercept)]);
disp(['RescaleSlope : ' num2str(RescaleSlope)]);
disp(['interslicegap : ' num2str(interslicegap)]);
disp(['user_def2 : ' num2str(user_def2)]);
disp(['par2 : ' num2str(par2)]);
disp(['offset : ' num2str(offset)]);


fout=fopen(fname,'wb','ieee-be');
fwrite(fout,uint16(sizes),'ushort'); % 4
fwrite(fout,uint16(image_type),'ushort'); % 1
fwrite(fout,single(scales),'float'); % 4
fwrite(fout,patient,'char'); % 80
fwrite(fout,single(matrix),'float'); % 20
fwrite(fout,uint8(orientation),'uint8'); % 1
fwrite(fout,uint8(par2),'uint8'); % 1
fwrite(fout,double(voxmin),'double'); % 1
fwrite(fout,double(voxmax),'double'); % 1
fwrite(fout,double(origin),'double'); % 4
fwrite(fout,single(RescaleIntercept),'float'); % 1
fwrite(fout,single(RescaleSlope),'float'); % 1
fwrite(fout,single(interslicegap),'float'); % 1
fwrite(fout,single(user_def2),'float'); % 1
fwrite(fout,uint32(magic_number),'uint'); % 1
fwrite(fout,I, trans_type{image_type});
fclose('all');



