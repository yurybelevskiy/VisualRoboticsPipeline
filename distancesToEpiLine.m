function [ distance ] = distancesToEpiLine( F, p1, p2 )
% Function that computes the distance from the epipolar lines for each pair
% of points.
% Input : - F (3x3) : fundamental matrix;
%         - p1(3xN) : homogeneous coordinates of points on frame 1;
%         - p2(3xN) : homogeneous coordinates of points on frame 2;
%Output : - distance (Nx1) : vector containing the squared distance of each
%                            pair of points from the epipolar lines.

homog_points = [p1, p2];
epi_lines = [F.'*p2, F*p1];


distance = (sum(epi_lines.*homog_points,1)).^2;

end

