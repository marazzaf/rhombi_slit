L = 16;
H = 16;
h = 0.1; //mesh size
l = 0.5;

Point(1) = {-L/2, 0, 0, h};
Point(2) = {L/2, 0, 0, h};
Point(3) = {L/2, H, 0, h};
Point(4) = {-L/2, H, 0, h};
Point(5) = {-L/2, H/2-l, 0, h};
Point(6) = {-L/2, H/2+l, 0, h};
Point(7) = {L/2, H/2-l, 0, h};
Point(8) = {L/2, H/2+l, 0, h};

Line(1) = {1,2};
Line(2) = {2,7};
Line(3) = {7,8};
Line(4) = {8,3};
Line(5) = {3,4};
Line(6) = {4,6};
Line(7) = {6,5};
Line(8) = {5,1};

Line Loop(1) = {1:8};

Plane Surface(1) = {1};

Physical Line(1) = {8,1,2,4,5,6};
Physical Line(2) = {3,7};

Physical Surface(1) = {1};