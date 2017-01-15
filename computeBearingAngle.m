function [ bear_angle ] = computeBearingAngle( p1, p2, K_1, K_2, pose_1_mat, pose_2_mat )
% Function that computes the normalized bearing vectors for 2 sets of 
% points and gives back the angle between the two bearing vectors created
% from the points given as inputs.

n1 = size(p1,2);

bear_angle = zeros(n1,1);

for ii = 1:n1
    pose_1 = reshape(pose_1_mat(:,ii),3,4); 
    pose_2 = reshape(pose_2_mat(:,ii),3,4); 

    bearings_1 = [pose_1; zeros(1,3),1]\[K_1\[p1(:,ii); 1]; 1];
    bearings_1 = bearings_1(1:3);
    
    bearings_2 = [pose_2; zeros(1,3),1]\[K_2\[p2(:,ii); 1]; 1];
    bearings_2 = bearings_2(1:3);
    
    bear_angle(ii) = asin(norm(cross(bearings_1,bearings_2),2)./(...
        norm(bearings_1,2)*norm(bearings_2,2)))*180/pi;
end

end

