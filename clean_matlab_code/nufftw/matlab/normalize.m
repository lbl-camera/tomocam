function nd = normalize(data, ref)

nd = data / maxabs(data);
if nargin > 1
    nd = nd * maxabs(ref);
end