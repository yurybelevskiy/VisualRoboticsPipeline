function [new_state, W_T_c] = processFrame(old_state, old_frame, new_frame, K)
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

plot_coordinate = false;

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
if plot_coordinate
    figure(4); clf; imshow(old_frame); hold on;
    plot(old_keypoints(1,:),old_keypoints(2,:),'rx','linewidth',1.75);
    plot(new_keypoints(1,:),new_keypoints(2,:),'go','linewidth',1.75);
    hold off;
end

% 2 - Camera pose estimation with P3P and RANSAC 
[R_C_W, t_C_W, inlier_mask] = ourRansacLocalization(new_keypoints,...
    p_W_landmarks,K);
new_keypoints = new_keypoints(:,inlier_mask>0);
p_W_landmarks = p_W_landmarks(:,inlier_mask>0);

% Display movement of the coordinate frame - just debugging purposes
if plot_coordinate
    figure(6); clf;
    plotCoordinateFrame(eye(3),zeros(3,1), 0.8);
    text(-0.1,-0.1,-0.1,'Cam 1','fontsize',10,'color','k','FontWeight','bold');
    center_cam2_W = -R_C_W'*t_C_W;
    plotCoordinateFrame(R_C_W',center_cam2_W, 0.8);
    text(center_cam2_W(1)-0.1, center_cam2_W(2)-0.1, center_cam2_W(3)-0.1,'Cam 2','fontsize',10,'color','k','FontWeight','bold');
    rotate3d on; grid on;
end

% 3 - Triangulating new landmarks
% Initialization of new candidate keypoints
% First of all, we need to create the match between the points we have in
% the old state and the new ones which come from the KLT.
% If we can track a point, we store the FIRST TRACKED KEYPOINT and it will
% be discarded when we cannot continue the tracking in the future
% iterations (see pipeline, section 4.2).

% Put in the state only if the candidate keypoint is the first time.
% Discard from the state what cannot be tracked anymore.
c_m_old = old_state(6:7,:); % First track of the keypoints

% Initialization step 3
harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 300; % <---------- TO BE TUNED
nonmaximum_supression_radius = 6;
descriptor_radius = 9;
match_lambda = 6; % <---------- TO BE TUNED PROPERLY FOR THE DESCRIPTOR MATCHING!!!!

scores_m_new = harris(new_frame,harris_patch_size,harris_kappa);
c_m_new = selectKeypoints(...
    scores_m_new, num_keypoints, nonmaximum_supression_radius);

% Track c_m_(i-1) in the frame I_i. Use descriptor matching (COULD WE USE
% KLT? ANY IDEA?) and take only the ones that are matched.
descriptors_m_new = describeKeypoints(new_frame,c_m_new,descriptor_radius);
descriptors_m_old = describeKeypoints(old_frame,c_m_old,descriptor_radius);
matches = matchDescriptors(descriptors_m_new,descriptors_m_old,match_lambda);

% Keep only the matched ones (the old one, since we want to keep in memory
% the fisrt track - mantain Markov property)
c_m_old = c_m_old(:,matches(matches>0)); % COME FACCIO A INSERIRE I NUOVI PUNTI?

% Intermediate output for part 1&2
W_T_c = [R_C_W, t_C_W];
M = repmat(reshape([W_T_c; 0,0,0,1],16,1),1,size(new_keypoints,2));
new_state = [new_keypoints; p_W_landmarks; c_m_old; M];
end

