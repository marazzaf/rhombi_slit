L = 1.5;
H = 1.5;
h = 0.005; //mesh size

Point(1) = {-L/2, -H/2, 0, h};
Point(2) = {L/2, -H/2, 0, h};
Point(3) = {L/2, H/2, 0, h};
Point(4) = {-L/2, H/2, 0, h};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line Loop(1) = {1:4};

Plane Surface(1) = {1};

Physical Line(1) = {1,3};
Physical Line(2) = {2,4};

Physical Surface(1) = {1};