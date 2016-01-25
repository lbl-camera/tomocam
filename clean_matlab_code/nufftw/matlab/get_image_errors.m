function [a,b] = get_image_errors(image1, image2)


a = abs(normalize(image1) - normalize(image2));
b = abs((image1-image2)/maxabs(image1));

disp(['max error: ' num2str(max(a(:)))])

% m = max(a(:));
% n = max(b(:));
% k = mean(a(:));
% r = mean(b(:));
% w = std(a(:));
% y = std(b(:));
% 
% str = sprintf('\t both normalized to 1 \t difference normalized to 1');
% disp(str)
% str = sprintf('max: \t %g \t %g',m,n);
% disp(str)
% str = sprintf('mean: \t %g \t %g',k,r);
% disp(str)
% str = sprintf('std: \t %g \t %g\n',w,y);
% disp(str)
