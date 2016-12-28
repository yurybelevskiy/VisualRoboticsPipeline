function [ p_new, pose_concatenated ] = computeMovement( new_pose, old_pose, p_old )
% Function to compute the new position to be compared with the ground
% truth. It uses the camera pose and homogenouse coordinates.
% The pose is composed as follows: pose = [R|t].

% If it has not homogenous coordinates, create them.
if size(p_old,1) < 4
    p_old = [p_old; ones(1,size(p_old,2))];
end

% Creation of the transformation matrix if not yet in the correct format
n_new = size(new_pose,1);
n_old = size(old_pose,1);

if n_new ~=4 
    new_pose = [new_pose; zeros(1,3), 1];
end

if n_old ~= 4
    old_pose = [old_pose; zeros(1,3), 1];
end

M = old_pose*new_pose;

% Extract the new pose
pose_concatenated = M(1:3,:);

% % Compute the new position (homogenous coordinates)
% p_new = M*p_old;
% % Extract (x,y,z) of the new position
% p_new = p_new(1:3);

p_new = M(1:3,end);

end

