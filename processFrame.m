function [new_state, W_T_c] = processFrame(old_state, old_frame, new_frame, K)
% Function for the continuos VO operation. It takes as inputs the state for
% the previous frame plus the previous and the new frames. 
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
debug_triangulation = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 - Point tracking : KLT Algorihm with pyramidal scheme (3 levels)
% % old_keypoints = old_state(1:2,:);
% % p_W_landmarks = old_state(3:5,:);
stop = find(isnan(old_state(:,3:5)),1,'first')-1;
old_keypoints = old_state(1:stop,1:2)';
p_W_landmarks = old_state(1:stop,3:5)';

% Transfor the type of the variable containing the keypoints
old_points = cornerPoints(old_keypoints');
tracker = vision.PointTracker('NumPyramidLevels',3,'MaxBidirectionalError',1,'MaxIterations',100);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    line([0,center_cam2_W(1)],[0,center_cam2_W(2)],[0,center_cam2_W(3)]);
end

% Partial output computation
W_T_c = [R_C_W, t_C_W];
M_new = repmat(reshape(W_T_c,12,1),1,size(new_keypoints,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3 - Triangulating new landmarks
% Initialization of new candidate keypoints
% First of all, we need to create the match between the points we have in
% the old state and the new ones which come from the KLT.
% If we can track a point, we store the FIRST TRACKED KEYPOINT and it will
% be discarded when we cannot continue the tracking in the future
% iterations (see pipeline, section 4.2).

% Put in the state only if the candidate keypoint is the first time.
% Discard from the state what cannot be tracked anymore.

% Creation of the temporary state
% Extract the 2D points + pose
p_old = old_state(stop+1:end,1:2)';
pose_old = old_state(stop+1:end,6:end)';
% Compute new keypoints in frame I_current
harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 300; % <---------- TO BE TUNED
nonmaximum_supression_radius = 6;
descriptor_radius = 9;
match_lambda = 6; % <---------- TO BE TUNED PROPERLY FOR THE DESCRIPTOR MATCHING!!!!
scores_m_new = harris(new_frame,harris_patch_size,harris_kappa);
p_new = selectKeypoints(...
    scores_m_new, num_keypoints, nonmaximum_supression_radius);
% Match the descriptors
descriptors_m_new = describeKeypoints(new_frame,p_new,descriptor_radius);
descriptors_m_old = describeKeypoints(old_frame,flipud(p_old),descriptor_radius);
matches = matchDescriptors(descriptors_m_new,descriptors_m_old,match_lambda);
% Divide the new candidate keypoints into matched and not-matched
matched_new = flipud(p_new(:,matches>0));
not_matched_new = flipud(p_new(:,matches==0));
% Old points that have been tracked
matched_old = p_old(:,matches(matches>0));
matched_pose_old = pose_old(:,matches(matches>0));
% State
matched_pose_new = repmat(reshape(W_T_c,12,1),1,size(matched_new,2));
not_matched_pose_new = repmat(reshape(W_T_c,12,1),1,size(not_matched_new,2));
% state_temp = [new_keypoints',p_W_landmarks',M_new';
%     matched_old', NaN(size(matched_old,2),3), pose_old';
%     matched_new', NaN(size(matched_new,2),3), matched_pose_new';
%     not_matched_new', NaN(size(not_matched_new,2),3), not_matched_pose_new'];

% Check traingulation
angle_tolerance = 2;
angles = computeBearingAngle(matched_new,matched_old,K);
if nnz(angles > angle_tolerance) ~= 0
    % Find the point pairs that can be triangulated
    idx = angles > angle_tolerance;
    tri_matched_old = matched_old(:,idx);
    tri_matched_new = matched_new(:,idx); 
    tri_matched_pose_old = matched_pose_old(:,idx);
    tri_matched_pose_new = matched_pose_new(:,idx);
    tri_p_W = zeros(4,size(tri_matched_pose_old,2));
    tri_matched_new_h = [tri_matched_new; ones(1,size(tri_matched_new,2))]; % Homogenous coordinates
    tri_matched_old_h = [tri_matched_old; ones(1,size(tri_matched_old,2))]; % Homogenous coordinates
    
    for i = 1:size(tri_matched_pose_old,2)
        tri_pose_old_temp = K*reshape(tri_matched_pose_old(:,i),3,4);%%%%%%%%%%
        tri_pose_new_temp = K*reshape(tri_matched_pose_new(:,i),3,4);%%%%%%%%%%
        tri_p_W(:,i) = linearTriangulation(tri_matched_old_h(:,i),...
            tri_matched_new_h(:,i),tri_pose_old_temp,tri_pose_new_temp);
    end
    % Discard negative depth
    tri_p_W = tri_p_W(1:3,:); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% no!!!!!!!!!
    idx = find(tri_p_W(3,:)>0,1);
    tri_p_W = tri_p_W(:,idx);
    tri_matched_new = tri_matched_new(:,idx);
    tri_matched_pose_new = tri_matched_pose_new(:,idx);
    
    new_state = [new_keypoints',p_W_landmarks',M_new';
        tri_matched_new', tri_p_W', tri_matched_pose_new';
        matched_old', NaN(size(matched_pose_old,2),3), matched_pose_old';
        not_matched_new', NaN(size(not_matched_new,2),3), not_matched_pose_new'];
else
    new_state = [new_keypoints',p_W_landmarks',M_new';
        matched_old', NaN(size(matched_pose_old,2),3), matched_pose_old';
        not_matched_new', NaN(size(not_matched_new,2),3), not_matched_pose_new'];
end
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Intermediate output for part 1&2
% M_new = M_new';
% M_new_m = repmat(reshape(W_T_c,12,1),1,size(p_new,2))';
% new_state = [new_keypoints', p_W_landmarks', M_new;
%     c_m, -ones(size(c_m,1),3), [M_old; M_new_m]];
end


