function [new_state, pose] = processFrame(old_state, old_frame, new_frame,K)
% Function for the continuos VO operation. It takes as inputs the state for
% the previous frame and the previous and the new frames. 
% The outputs of the function are the camera pose and the new state.
%
% Input:    - old_state : state that has the following structure:
%                         [2D keypoints; 3D landmarks; descriptors];
%           - old_frame : previous frame corresponding to old_state;
%           - new_frame : new frame whose pose and state are computed.
% 
% Output:   - new_state : new state with the same structure as old_state;
%           - pose      : pose of the camera for the new frame.

% The function is divided into two main parts: 
%   1) point traking with Lucas-Kanade tracker;
%   2) pose estimation (with RANSAC).

% 1 - Point tracking : KLT Algorihm with pyramidal scheme (3 levels)
old_keypoints = old_state(1:2,:);
p_W_landmarks = old_state(3:5,:);

% Transfor the type of the variable containing the keypoints
old_points = cornerPoints(old_keypoints');
tracker = vision.PointTracker('NumPyramidLevels',3,'MaxBidirectionalError',1);
initialize(tracker, old_points.Location, old_frame);
[new_points, validity] = step(tracker, new_frame);
% Update the correspondence 2D - 3D
new_points = double(new_points(validity>0,:)');
p_W_landmarks = p_W_landmarks(:,validity>0);

% 2 - Camera pose estimation with P3P and RANSAC
% [R_C_W, t_C_W, inlier_mask] = ourRansacLocalization(new_points,...
%     p_W_landmarks,K);
% new_points = new_points(:,inlier_mask>0);
% p_W_landmarks = p_W_landmarks(:,inlier_mask>0);

% Final output
new_state = [new_points; p_W_landmarks]; 
pose = 0;%[R_C_W, t_C_W];

end

