function image = read_image(filename,format)

if nargin < 2
    format = 'double';
end
image = vec2complex(readbin(filename,format));
s = sqrt(length(image));
image = reshape(image,s,s);