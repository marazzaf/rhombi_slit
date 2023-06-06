L = 15;
H = 15;
h = 0.05; //mesh size
eps = 0.1;

Point(1) = {0+eps, 0+eps, 0, h};
Point(2) = {L-eps, 0+eps, 0, h};
Point(3) = {L-eps, H-eps, 0, h};
Point(4) = {0+eps, H-eps, 0, h};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line Loop(1) = {1:4};

Plane Surface(1) = {1};

Physical Line(1) = {1,3};
Physical Line(2) = {2,4};

Physical Surface(1) = {1};