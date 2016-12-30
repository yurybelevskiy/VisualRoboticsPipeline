function [ bear_angle ] = computeBearingAngle( p1, p2, K )
% Function that computes the normalized bearing vectors for 2 sets of 
% points and gives back the angle between the two bearing vectors created
% from the points given as inputs.

n1 = size(p1,2);
n2 = size(p2,2);

norm_bearings_1 = K\[p1; ones(1, n1)];
norm_bearings_2 = K\[p2; ones(1, n2)];
bear_angle = zeros(n1,1);

for ii = 1:n1
    norm_bearings_1(:, ii) = norm_bearings_1(:, ii) / ...
        norm(norm_bearings_1(:, ii), 2);
    norm_bearings_2(:, ii) = norm_bearings_2(:, ii) / ...
        norm(norm_bearings_2(:, ii), 2);
    bear_angle(ii) = asin(norm(cross(norm_bearings_1,norm_bearings_2))./(...
        norm(norm_bearings_1)*norm(norm_bearings_2)))*180/pi;
end

end

