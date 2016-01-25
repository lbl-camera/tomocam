function h = huber_pot(t, d)
% huber potential function
h = t.^2 / 2;
ii = abs(t) > d;
h(ii) = d * abs(t(ii)) - d.^2/2;
