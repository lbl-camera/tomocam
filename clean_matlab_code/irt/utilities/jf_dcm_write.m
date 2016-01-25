function result = jf_dcm_write(var, dcm_name, ...
	WindowCenter, WindowWidth, ...
	RescaleIntercept, RescaleSlope, ...
	PixelSpacing, ImageOrientationPatient, ...
	ImagePositionPatient, ImageType)
% usage:
% fld2dcm(fld_name, dcm_name, WindowCenter, WindowWidth)
%
% for 2d CT images only

if (nargin <= 1) % run test routine

	% peaks.fld created using the following code:-
	%temp=single(1000 + 30*peaks(512));
	%temp = temp';
	%fld_write('peaks.fld',temp);
	%im(temp,[800 1200])
	%imshow(temp',[800 1200])
	% a dcm viewer should show an image similar to the ones displayed above

	result = fld2dcm('peaks.fld', '/tmp/peaks.dcm', 1000, 400);
return
end

if (nargin < 4)
	result = -3;
	fprintf('not enough arguments. quitting.\n')
	return
end

% the last 4 options below are required, but seem not to be important
% for display purposes; they can be anything for now.
if ~exist('RescaleIntercept','var')
	RescaleIntercept = 0;
end
if ~exist('RescaleSlope','var')
	RescaleSlope = 1;
end
if ~exist('PixelSpacing','var')
	PixelSpacing = [1 1] * 700.3125/512;
end
if ~exist('ImageOrientationPatient','var')
	ImageOrientationPatient = [1 0 0 0 1 0];
end
if ~exist('ImagePositionPatient','var')
	ImagePositionPatient = [-250.0000 -250.0000 -57.8750];
end
if ~exist('ImageType','var')
	ImageType = 'ORIGINAL\SECONDARY\AXIAL';
end

result = 0; % success

tmp_file = '/tmp/fld2dcm_temp.dcm';

data_in = fld_read(fld_name,'raw',1);
data_in = int16(data_in);
data_in = permute(data_in,[2 1]); % transpose in 2d

% write out a dummy dcm file. this lacks required fields.
status = dicomwrite(data_in, tmp_file, 'ObjectType', 'CT Image Storage');

if (length(status.MissingData) ~= 0)
	fprintf('inserting missing metadata ...');

	metadata = dicominfo(tmp_file);
	delete(tmp_file);

	metadata.WindowCenter	= WindowCenter;
	metadata.WindowWidth	= WindowWidth;
	metadata.RescaleIntercept = RescaleIntercept;
	metadata.RescaleSlope	= RescaleSlope;
	metadata.PixelSpacing	= PixelSpacing;
	metadata.ImageOrientationPatient = ImageOrientationPatient;
	metadata.ImagePositionPatient	= ImagePositionPatient;
	metadata.ImageType	= ImageType;
	
	status = dicomwrite(data_in, dcm_name, metadata, 'ObjectType', 'CT Image Storage');
	if (length(status.MissingData) ~= 0)
		fprintf('failed. quitting.\n');
		result = -1;
		return;
	else
		fprintf('done.\n');
	end
else
	fprintf('succeded unexpectedly. quitting.\n');
	delete(tmp_file);
	result = -2;
	return;
end


