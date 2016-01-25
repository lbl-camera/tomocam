function out = readbin(filename,format)


if nargin < 2
    format = 'double';
end
fid = fopen(filename,'rb');
out = fread(fid,format);
fclose(fid);