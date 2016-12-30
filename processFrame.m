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
stop = find(old_state(:,3:5)==-1,1,'first')-1;
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

% c_m_old = old_state(6:7,:)'; % track
% old_pose = old_state(8:end,:);
c_m_old = old_state(stop+1:end,1:2)'; % track
old_pose = old_state(stop+1:end,6:end)';

% Initialization step 3. 
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
descriptors_m_old = describeKeypoints(old_frame,flipud(c_m_old),descriptor_radius);
matches = matchDescriptors(descriptors_m_new,descriptors_m_old,match_lambda);

% Keep only the matched ones (the old one, since we want to keep in memory
% the fisrt track - mantain Markov property)
c_m_old = c_m_old(:,matches(matches>0)); % old points - the start of the tracks
M_old = old_pose(:,matches(matches>0))'; % correspondig poses of the initialization of tracks

c_m_new_tri = c_m_new(:,matches>0); % points with valid matching

if debug_triangulation
    plotMatched(old_frame, flipud(c_m_old), c_m_new_tri, false);
end

% Choose only the points that can be triangulated
angle_threshold = 1; % [°] how much???
% angles = computeBearingAngle(c_m_old,c_m_new(:,matches>0),K);
angles = computeBearingAngle(flipud(c_m_new_tri),c_m_old,K); %%
M_triang_new = [R_C_W, t_C_W]; % this one is fixed, since it is the last step. It is the pose of last iteration.

% Check if traingulation is possible
if nnz(angles>angle_threshold) ~= 0
    point_triang_old = c_m_old(:,angles>angle_threshold);
    point_triang_new = c_m_new_tri(:,angles>angle_threshold);
    M_triang_old = M_old(angles>angle_threshold,:);
    key_h1 = [point_triang_old; ones(1,size(point_triang_old,2))];
    key_h2 = [flipud(point_triang_new); ones(1,size(point_triang_new,2))];
    p_W_landmarks_new = zeros(size(point_triang_old,2),4);
    for i = 1:size(point_triang_old,2)
        M1 = K*reshape(M_triang_old(i,:),3,4); % moltiplicare per K?!?
        M2 = K*M_triang_new;
        p_W_landmarks_new(i,:) = linearTriangulation(key_h1(:,i),key_h2(:,i),M1,M2)';
    end
    idx = p_W_landmarks_new(:,3)>0;
    p_W_landmarks_new = p_W_landmarks_new(idx,1:3);
    c_m_new_tri = c_m_new_tri(:,idx);
    c_m_old(:,angles>angle_threshold) = [];
    M_old(angles>angle_threshold,:) = [];
    
    % to do: cancellare i punti triangolati
    % cancellare z negative
    % cancellare i punti 2d
    
    c_m_new = c_m_new(:,matches==0); % Correspondence now is lost
    c_m = [c_m_old'; c_m_new'];
    M_new = M_new';
    M_new_m = repmat(reshape(W_T_c,12,1),1,size(c_m_new,2))';
    new_state = [new_keypoints', p_W_landmarks', M_new;
        c_m_new_tri', p_W_landmarks_new, M_new(1:size(c_m_new_tri,2),:);
        c_m, -ones(size(c_m,1),3), [M_old; M_new_m]];
else

% The other points are not tracked. However, we can keep the in memory. In
% this way we can see if they can be tracked in the successive iteration.
% If they are not tracked in the next step, they will be discarded.
c_m_new = c_m_new(:,matches==0); % Correspondence now is lost

% RANSAC? THEY WILL CONTAIN OUTLIERS
c_m = [c_m_old'; c_m_new'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Intermediate output for part 1&2
M_new = M_new';
M_new_m = repmat(reshape(W_T_c,12,1),1,size(c_m_new,2))';
new_state = [new_keypoints', p_W_landmarks', M_new;
    c_m, -ones(size(c_m,1),3), [M_old; M_new_m]];
end

end

