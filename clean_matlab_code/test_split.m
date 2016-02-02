% There is a problem with non square frames!
% But square frames seems to work

x = complex(single(zeros(3,3,2)));
y = single(rand(5,5) + i*rand(5,5)); 
frame_corner = fastlab(int32([0,5]));
frame = fastlab(x);
image = fastlab(y);
image_split(image,frame,frame_corner);
