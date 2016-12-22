function [new_state, W_T_c] = processFrame(old_state, old_frame, new_frame,K)
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
[new_keypoints, validity] = step(tracker, new_frame);
% Update the correspondence 2D - 3D
new_keypoints = double(new_keypoints(validity>0,:)');
p_W_landmarks = p_W_landmarks(:,validity>0);

% Plot the old keypoints and the new ones in the old frame to check
% process.
figure(1); clf; imshow(old_frame); hold on;
plot(old_keypoints(1,:),old_keypoints(2,:),'rx','linewidth',1.75);
plot(new_keypoints(1,:),new_keypoints(2,:),'go','linewidth',1.75);
hold off;

% 2 - Camera pose estimation with P3P and RANSAC
% % % Invert index from MatLab to our convenction
% % new_points = [new_points(2,:); new_points(1,:)]; 
[R_C_W, t_C_W, inlier_mask] = ourRansacLocalization(new_keypoints,...
    p_W_landmarks,K);
new_keypoints = new_keypoints(:,inlier_mask>0);
p_W_landmarks = p_W_landmarks(:,inlier_mask>0);

% 3 - Triangulating new landmarks
% Initialization of new candidate keypoints
% First of all, we need to create the match between the points we have in
% the old state and the new ones which come from the KLT.
% If we can track a point, we store the FIRST TRACKED KEYPOINT and it will
% be discarded when we cannot continue the tracking in the future
% iterations (see pipeline, section 4.2).

% Put in the state only if the candidate keypoint is the first time.
% Discard from the state what cannot be tracked anymore.
candidate_triangulation = old_state(6:7,:);
% Take only the tracked and the matched ones
candidate_triangulation = candidate_triangulation(:,validity>0); 
candidate_triangulation = candidate_triangulation(:,inlier_mask>0);

% Intermediate output for part 1&2
W_T_c = [R_C_W, t_C_W];
M = repmat(reshape([W_T_c; 0,0,0,1],16,1),1,size(new_keypoints,2));
new_state = [new_keypoints; p_W_landmarks; candidate_triangulation; M]; 

end

