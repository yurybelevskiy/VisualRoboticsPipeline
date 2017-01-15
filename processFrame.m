function [new_state, W_T_c, error] = processFrame(old_state, old_frame, new_frame, K, frame_n)
% Function for the continuos VO operation. It takes as inputs the state for
% the previous frame plus the previous and the new frames.
% The outputs of the function are the camera pose and the new state.
%
% Input:    - old_state : state that has the following structure:
%                         [2D keypoints; 3D landmarks; descriptors];
%           - old_frame : previous frame corresponding to old_state;
%           - new_frame : new frame whose pose and state are computed;
%           - frame_n   : current frame.
%
% Output:   - new_state : new state with the same structure as old_state;
%           - pose      : pose of the camera for the new frame;
%           - error     : if an error occured, this value is negative.

% The function is divided into three main parts:
%   1) point traking with Lucas-Kanade tracker;
%   2) pose estimation (with RANSAC);
%   3) new landmarks triangulation.

% Parameter initialization
num_not_matched_new = 200;
num_not_matched_new_emerg = 400;

%% 1 - Point tracking : KLT Algorihm with pyramidal scheme (3 levels)
stop = find(isnan(old_state(:,5)),1,'first')-1;
[H,W] = size(old_frame);

if ~isempty(stop)
    old_keypoints = old_state(1:stop,1:2)'; % 2D points to track
    p_W_landmarks = old_state(1:stop,3:5)'; % Corresponding 3D points
    
    % Delete not valid points
    to_be_deleted = old_keypoints(1,:)<0 | old_keypoints(2,:)<0 | ...
        old_keypoints(1,:)>W | old_keypoints(2,:)>H;
    old_keypoints(:,to_be_deleted) = [];
    p_W_landmarks(:,to_be_deleted) = [];
else
    old_keypoints = old_state(:,1:2)';
    p_W_landmarks = old_state(:,3:5)';
    to_be_deleted = old_keypoints(1,:)<0 | old_keypoints(2,:)<0 | ...
        old_keypoints(1,:)>W | old_keypoints(2,:)>H;
    old_keypoints(:,to_be_deleted) = [];
    p_W_landmarks(:,to_be_deleted) = [];
end

% Transform the type of the variable containing the keypoints
old_points = cornerPoints(old_keypoints');
tracker = vision.PointTracker('NumPyramidLevels',3,'MaxBidirectionalError',1);
initialize(tracker, old_points.Location, old_frame);
[new_keypoints, validity] = step(tracker, new_frame);

% Update the correspondence 2D - 3D
new_keypoints = double(new_keypoints(validity>0,:)');

if (isempty(new_keypoints) | validity==zeros(size(validity)))
    disp('No points from the previous frame have been tracked.');
    new_keypoints = old_keypoints;
else
    p_W_landmarks = p_W_landmarks(:,validity>0);
end
    
%% 2 - Camera pose estimation with P3P and RANSAC
if size(new_keypoints,2) > 3
    pixel_tolerance = 10;
    [R_C_W, t_C_W, inlier_mask] = ourRansacLocalization(new_keypoints,...
        p_W_landmarks,K,pixel_tolerance);
    
    new_keypoints = new_keypoints(:,inlier_mask>0);
    p_W_landmarks = p_W_landmarks(:,inlier_mask>0);
    
    % Pose of the current step
    W_T_c = [R_C_W, t_C_W];
    
    if isempty(W_T_c)
        % Compute new keypoints in frame I_current
        harris_patch_size = 9;
        harris_kappa = 0.08;
        num_keypoints = 800;
        nonmaximum_supression_radius = 6;
        
        scores_old_frame = harris(old_frame,harris_patch_size,harris_kappa);
        p_old_frame = selectKeypoints(...
            scores_old_frame, num_keypoints, nonmaximum_supression_radius);
        old_points = cornerPoints(p_old_frame');
        tracker = vision.PointTracker('NumPyramidLevels',3);
        initialize(tracker, old_points.Location, old_frame);
        [new_keypoints, validity] = step(tracker, new_frame);
        % Update the correspondence 2D - 3D
        new_keypoints = double(new_keypoints(validity>0,:)');
    else
        M_new = repmat(reshape(W_T_c,12,1),1,size(new_keypoints,2));
    end
end

%% 3 - Triangulating new landmarks
% First of all, we need to create the match between the points we have in
% the old state and the new ones which come from Lukas-Kanade.
% If we can track a point, we store the FIRST TRACKED KEYPOINT and it will
% be discarded when we cannot continue the tracking in the future
% iterations (see pipeline, section 4.2).

% Try to track the points that were not matched in the previous step
stop_2 = find(isnan(old_state(:,3)),1,'first')-1;
p_old = old_state(stop+1:end,1:2)'; 
if ~isempty(stop_2)
    p_old_last = [old_state(stop+1:stop_2,3:4)', ... % latest coordinates of tracked keypoints
        old_state(stop_2+1:end,1:2)'];               % coordinates from previous state.
else
    p_old_last = old_state(stop+1:end,3:4)';
end
pose_old = old_state(stop+1:end,6:end)'; % pose from previous steps.

% Track of the points to be triangulated
old_points = cornerPoints(p_old_last');
tracker = vision.PointTracker('NumPyramidLevels',3,'MaxBidirectionalError',1);
initialize(tracker, old_points.Location, old_frame);
[p_new, matches] = step(tracker,new_frame);

% Discard wrong tracking
p_new = double(p_new');
to_be_deleted = p_new(1,:)<0 | p_new(2,:)<0 | p_new(1,:)>W | p_new(2,:)>H;
p_new(:,to_be_deleted) = [];
matches(to_be_deleted) = [];
p_old(:,to_be_deleted) = [];
pose_old(:,to_be_deleted) = [];

% Extract the right matches
matched_old = p_old(:,matches>0);
matched_old_last = p_old_last(:,matches>0);
matched_new = p_new(:,matches>0); 
matched_pose_old = pose_old(:,matches>0);

% Extract new points to be triangulated in the next steps
not_matched_new = detectHarrisFeatures(new_frame,'MinQuality',0.1);
% Emergency check: if we have a too low number of point correspondences,
% enlarge the pool of possible triangulable points.
if size(new_keypoints,2)>50
    not_matched_new = not_matched_new.selectStrongest(num_not_matched_new);
else
    not_matched_new = not_matched_new.selectStrongest(num_not_matched_new_emerg);
end

not_matched_new = double(not_matched_new.Location');
to_be_deleted = not_matched_new(1,:)<0 | not_matched_new(2,:)<0 | ...
    not_matched_new(1,:)>W | not_matched_new(2,:)>H;
not_matched_new(:,to_be_deleted) = [];
   
% Creation of the poses for all the points
if exist('W_T_c','var') && ~isempty(W_T_c)
    matched_pose_new = repmat(reshape(W_T_c,12,1),1,size(matched_new,2));
    not_matched_pose_new = repmat(reshape(W_T_c,12,1),1,size(not_matched_new,2));
else
    pose_temp = old_state(1,6:end)';
    matched_pose_new = repmat(pose_temp,1,size(matched_new,2));
    not_matched_pose_new = repmat(pose_temp,1,size(not_matched_new,2));
end

% Traingulation check: points can be triangulated if the angle between the
% bearing vector is higher than a certain threshold. The bearing vectors 
% are expressed in the wolrd frame.

if frame_n < 4
    angle_threshold = 5;        % Only for the first frames
elseif size(new_keypoints,2)<40 % Emergency check
    angle_threshold = 0.4;
else
    angle_threshold = 1;
end     

% Compute the angles between bearing vectors
angles = computeBearingAngle(matched_new,matched_old_last,K,K,matched_pose_new,matched_pose_old); 

% Print message - debugging
fprintf('Number of tracked points: %d over %d;\n',size(new_keypoints,2),size(old_keypoints,2));

if nnz(angles > angle_threshold) ~= 0
    % Find the point pairs that can be triangulated
    idx = angles > angle_threshold;
    
    % Extract the points from previous steps that can be triangulated
    tri_matched_old = matched_old(:,idx);
    
    % Keep track of the points from previous steps that have been tracked,
    % but that cannot be triangulated.
    matched_old = matched_old(:,1-idx>0); % Take only the ones that cannot be triangulated
    
    % Extract the points from step k that have been tracked and that can be
    % triangulated.
    tri_matched_new = matched_new(:,idx);
    matched_new = matched_new(:,1-idx>0);
    
    % Extract the corresponding poses.
    tri_matched_pose_old = matched_pose_old(:,idx); % Tracked + triangulable 
    matched_pose_old = matched_pose_old(:,1-idx>0); % Tracked NOT triangulable 
    tri_matched_pose_new = matched_pose_new(:,idx); % New tracked + triangulable
    
    % Initialization of the variables for computation speed.
    tri_p_W = zeros(4,size(tri_matched_pose_old,2));
    proj_p_W = zeros(3,size(tri_matched_pose_old,2));
    
    % Creation of the homogenous coordinates.
    tri_matched_new_h = [tri_matched_new; ones(1,size(tri_matched_new,2))]; 
    tri_matched_old_h = [tri_matched_old; ones(1,size(tri_matched_old,2))]; 
    
    % Iterate over the number of points that can be triangulated. A for
    % cycle is necessary since we have to pick the right pose from the
    % database.
    for i = 1:size(tri_matched_pose_old,2)
        % Extract the right pose.
        tri_pose_old_temp = reshape(tri_matched_pose_old(:,i),3,4);
        tri_pose_new_temp = reshape(tri_matched_pose_new(:,i),3,4);
        
        % Triangulation
        tri_p_W(:,i) = linearTriangulation(tri_matched_new_h(:,i),...
            tri_matched_old_h(:,i),K*tri_pose_new_temp,K*tri_pose_old_temp);
        % Triangulation check:
        % Projected point in the current camera reference frame: the
        % projected points with z<0 will be discarded.
        proj_p_W(:,i) = tri_pose_new_temp(1:3,1:3)*tri_p_W(1:3,i)+tri_pose_new_temp(1:3,4);
    end
    % Discard negative depth wrt the current camera pose
    idx = find(proj_p_W(3,:)>0 & proj_p_W(3,:)<400);
    tri_p_W = tri_p_W(1:3,idx);                         % Valid triangulated points
    tri_matched_new = tri_matched_new(:,idx);           % 2D position corresponding to the valid triangulated points
    tri_matched_pose_new = tri_matched_pose_new(:,idx); % Pose of the valid 2D-3D correspondences.

    % There are a lot of outliers; so apply ransac!!!
    if size(tri_matched_new,2) > 3 % Check if P3P can be run
        total_key = [new_keypoints, tri_matched_new];
        total_3d = [p_W_landmarks, tri_p_W];
        pixel_tolerance = 10;
        [R, t, inlier_mask] = ourRansacLocalization(total_key, total_3d, K, pixel_tolerance);
        if ~exist('R','var') || isempty([R t])
            error = -1;
            new_state = 0;
            W_T_c = [];
            return;
        end 
        % Extract the inliers
        inlier_mask_tri = inlier_mask(size(p_W_landmarks,2)+1:end);

        tri_matched_new = tri_matched_new(:,inlier_mask_tri>0);
        tri_matched_pose_new = tri_matched_pose_new(:,inlier_mask_tri>0);
        tri_p_W = tri_p_W(:,inlier_mask_tri>0);
        not_matched_pose_new = repmat(reshape([R,t],12,1),1,size(not_matched_new,2)); 
        W_T_c = [R,t];                       
        M_new = repmat(reshape(W_T_c,12,1),1,size(new_keypoints,2));
    end
    
    % Print message
    if exist('inlier_mask_tri','var')
        fprintf('Number of triangulated landmarks: %d.\n',nnz(inlier_mask_tri));
    else
        fprintf('Punti triangolati validi %d su %d\n\n', nnz(idx),size(tri_matched_new_h,2));
    end
      
    % Error check
    if ~exist('W_T_c','var') || isempty(W_T_c)
        error = -1; 
        new_state = 0;
        W_T_c = [];
        return;
    else
        error = 0;
    end
    
    % Creation of the new state
    new_state = [new_keypoints',p_W_landmarks',M_new';
        tri_matched_new', tri_p_W', tri_matched_pose_new';
        matched_old', matched_new', NaN(size(matched_pose_old,2),1), matched_pose_old';
        not_matched_new', NaN(size(not_matched_new,2),3), not_matched_pose_new'];
else
    % Error check
    if ~exist('W_T_c','var') || isempty(W_T_c)
        error = -1; 
        new_state = 0;
        W_T_c = [];
        return;
    else
        error = 0;
    end
    
    % Creation of the new state
    new_state = [new_keypoints',p_W_landmarks',M_new';
        matched_old', matched_new', NaN(size(matched_pose_old,2),1), matched_pose_old';
        not_matched_new', NaN(size(not_matched_new,2),3), not_matched_pose_new'];
end

end


