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

plot_coordinate = 0;
plot_coordinate_3d = 0;
debug_plot = 0;
debug_ransac = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 - Point tracking : KLT Algorihm with pyramidal scheme (3 levels)
stop = find(isnan(old_state(:,3:5)),1,'first')-1;
[H,W] = size(old_frame);

if ~isempty(stop)
    old_keypoints = old_state(1:stop,1:2)';
    p_W_landmarks = old_state(1:stop,3:5)';
    to_be_deleted = old_keypoints(1,:)<0 | old_keypoints(2,:)<0 | ...
        old_keypoints(1,:)>W | old_keypoints(2,:)>H;
    old_keypoints(:,to_be_deleted) = [];
else
    old_keypoints = old_state(:,1:2)';
    p_W_landmarks = old_state(:,3:5)';
    to_be_deleted = old_keypoints(1,:)<0 | old_keypoints(2,:)<0 | ...
        old_keypoints(1,:)<W | old_keypoints(2,:)<H;
    old_keypoints(:,to_be_deleted) = [];
end

% Transform the type of the variable containing the keypoints
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
    pause(0.1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2 - Camera pose estimation with P3P and RANSAC
if size(new_keypoints,2) ~= 0
    [R_C_W, t_C_W, inlier_mask] = ourRansacLocalization(new_keypoints,...
        p_W_landmarks,K);
end

new_keypoints = new_keypoints(:,inlier_mask>0);
p_W_landmarks = p_W_landmarks(:,inlier_mask>0);

% for plot only
% % old_keypoints = old_keypoints(:,validity>0);

% % old_keypoints = old_keypoints(:,inlier_mask>0);
% % plotMatched(new_frame,flipud(old_keypoints),flipud(new_keypoints),1,3);
% % pause(0.1);

% Display movement of the coordinate frame - just debugging purposes
if plot_coordinate_3d
    figure(6); clf;
    plotCoordinateFrame(eye(3),zeros(3,1), 0.8);
    text(-0.1,-0.1,-0.1,'Cam 1','fontsize',10,'color','k','FontWeight','bold');
    center_cam2_W = -R_C_W'*t_C_W;
    plotCoordinateFrame(R_C_W',center_cam2_W, 0.8);
    text(center_cam2_W(1)-0.1, center_cam2_W(2)-0.1, center_cam2_W(3)-0.1,...
        'Cam 2','fontsize',10,'color','k','FontWeight','bold');
    rotate3d on; grid on;
    line([0,center_cam2_W(1)],[0,center_cam2_W(2)],[0,center_cam2_W(3)]);
end

% Partial output computation
W_T_c = [R_C_W, t_C_W];
if size(new_keypoints,2) ~= 0
    M_new = repmat(reshape(W_T_c,12,1),1,size(new_keypoints,2));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3 - Triangulating new landmarks
% Initialization of new candidate keypoints
% First of all, we need to create the match between the points we have in
% the old state and the new ones which come from the KLT.
% If we can track a point, we store the FIRST TRACKED KEYPOINT and it will
% be discarded when we cannot continue the tracking in the future
% iterations (see pipeline, section 4.2).

% Try to track the points that were not matched in the previous step
p_old = old_state(stop+1:end,1:2)'; % traccia dei punti dai passi precedenti - punti iniziali della traccia
pose_old = old_state(stop+1:end,6:end)'; % pose delle tracce dei punti dai passi precedenti - pose iniziali della traccia

% Compute new keypoints in frame I_current
harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 1000; % <---------- TO BE TUNED 300
nonmaximum_supression_radius = 6;
descriptor_radius = 9;
match_lambda = 6; % <---------- TO BE TUNED PROPERLY FOR THE DESCRIPTOR MATCHING!!!!

scores_m_new = harris(new_frame,harris_patch_size,harris_kappa);
p_new = selectKeypoints(...
    scores_m_new, num_keypoints, nonmaximum_supression_radius);

%%%%%%%%%%%%%%%%%%%%
% p_new = detectHarrisFeatures(new_frame);
% p_new = flipud(p_new.Location');
% to_be_deleted = p_new(1,:)<0 | p_new(2,:)<0 | max(p_new(1,:))>H | max(p_new(2,:))>W;
% p_new(:,to_be_deleted) = [];
%%%%%%%%%%%%%%%%%%%%

% Match the descriptors
descriptors_m_new = ourDescribeKeypoints(new_frame,p_new,descriptor_radius);
descriptors_m_old = ourDescribeKeypoints(old_frame,flipud(p_old),descriptor_radius);
matches = matchDescriptors(descriptors_m_new,descriptors_m_old,match_lambda);

% Divide the new candidate keypoints into matched and not-matched
matched_new = flipud(p_new(:,matches>0)); % Punti identificati al passo k validi
not_matched_new = flipud(p_new(:,matches==0)); % Punti identificati al passo k NON validi
matched_old = p_old(:,matches(matches>0)); % Punti dal passo k-1 validi -> da tenerne traccia

% Creation of the poses for all the points
matched_pose_new = repmat(reshape(W_T_c,12,1),1,size(matched_new,2));
not_matched_pose_new = repmat(reshape(W_T_c,12,1),1,size(not_matched_new,2));
matched_pose_old = pose_old(:,matches(matches>0));

% Traingulation check: points can be triangulated if the angle between the
% bearing vector is higher than a certain threshold.
angle_threshold = 1;
angles = computeBearingAngle(matched_new,matched_old,K,K,matched_pose_new,matched_pose_old); 
% the bearing vectors are expressed in the wolrd frame.

% Print message - debugging
fprintf('Punti tracciati: %d su %d;\n',size(new_keypoints,2),size(old_keypoints,2));
X = ['         Max: ' num2str(max(angles(:))) ' with nnz: ' num2str(nnz(angles>angle_threshold)) ...
    ' out of ' num2str(length(angles)) '\n'];
fprintf(X);

if nnz(angles > angle_threshold) ~= 0
    % Find the point pairs that can be triangulated
    idx = angles > angle_threshold;
    
    % Extract the points from step k-1 that can be triangulated
    tri_matched_old = matched_old(:,idx);
    
    % Keep track of the points from previous steps that have been tracked,
    % but that cannot be triangulated.
    matched_old = matched_old(:,1-idx>0); % Take only the ones that cannot be triangulated
    
    % Extract the points from step k that have been tracked and that can be
    % triangulated.
    tri_matched_new = matched_new(:,idx);
    
    % Extract the corresponding poses.
    tri_matched_pose_old = matched_pose_old(:,idx); % Tracked + triangulable 
    matched_pose_old = matched_pose_old(:,1-idx>0); % Tracked NOT triangulable 
    tri_matched_pose_new = matched_pose_new(:,idx); % New tracked + triangulable
    
    % Initialization of the variables for computation speed.
    tri_p_W = zeros(4,size(tri_matched_pose_old,2));
    proj_p_W = zeros(3,size(tri_matched_pose_old,2));
    
    % Creation of the homogenous coordinates.
    tri_matched_new_h = [tri_matched_new; ones(1,size(tri_matched_new,2))]; % Homogenous coordinates
    tri_matched_old_h = [tri_matched_old; ones(1,size(tri_matched_old,2))]; % Homogenous coordinates
    
    % Iterate over the number of points that can be triangulated. A for
    % cycle is necessary since we have to pick the right pose from the
    % database.
    for i = 1:size(tri_matched_pose_old,2)
        % Extract the right pose.
        tri_pose_old_temp = reshape(tri_matched_pose_old(:,i),3,4);
        tri_pose_new_temp = reshape(tri_matched_pose_new(:,i),3,4);
        % Triangulation
        tri_p_W(:,i) = linearTriangulation(tri_matched_old_h(:,i),...
            tri_matched_new_h(:,i),K*tri_pose_old_temp,K*tri_pose_new_temp);
        % Triangulation check:
        % Projected point in the current camera reference frame: the
        % projected points with z<0 will be discarded.
        proj_p_W(:,i) = tri_pose_old_temp(1:3,1:3)*tri_p_W(1:3,i)+tri_pose_old_temp(1:3,4);
    end
    % Discard negative depth wrt the current camera pose
    idx = find(proj_p_W(3,:)>0);
    tri_p_W = tri_p_W(1:3,idx);     % Valid triangulated points
    tri_matched_new = tri_matched_new(:,idx); % 2D position corresponding to the valid triangulated points
    tri_matched_pose_new = tri_matched_pose_new(:,idx); % Pose of the valid 2D-3D correspondences.

    % There are a lot of outliers; so apply ransac!!!
    if size(tri_matched_new,2) > 3 % Check if P3P can be run
        [~, ~, inlier_mask_tri] = ourRansacLocalization(tri_matched_new,tri_p_W,K);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Plot - debugging only
        if debug_ransac
            tri_matched_old_plot = tri_matched_old(:,idx);
            figure(8);
            subplot(211)
            imshow(old_frame); hold on;
            plot(tri_matched_old_plot(1,:),tri_matched_old_plot(2,:),'rx','linewidth',1.5);
            plot(tri_matched_new(1,:),tri_matched_new(2,:),'go','linewidth',1.5);
            x_from = tri_matched_new(2,:); x_to = tri_matched_old_plot(2,:);
            y_from = tri_matched_new(1,:); y_to = tri_matched_old_plot(1,:);
            plot([y_from;y_to],[x_from;x_to],'y-','linewidth',1.25); hold off;
            title('Without ransac');
            
            subplot(212)
            imshow(old_frame); hold on;
            plot(tri_matched_old_plot(1,inlier_mask_tri>0),tri_matched_old_plot(2,inlier_mask_tri>0),'rx','linewidth',1.5);
            plot(tri_matched_new(1,inlier_mask_tri>0),tri_matched_new(2,inlier_mask_tri>0),'go','linewidth',1.5);
            x_from = tri_matched_new(2,inlier_mask_tri>0); x_to = tri_matched_old_plot(2,inlier_mask_tri>0);
            y_from = tri_matched_new(1,inlier_mask_tri>0); y_to = tri_matched_old_plot(1,inlier_mask_tri>0);
            plot([y_from;y_to],[x_from;x_to],'y-','linewidth',1.25); hold off;
            title('With ransac');
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Extract the inliers
        tri_matched_new = tri_matched_new(:,inlier_mask_tri>0);
        tri_matched_pose_new = tri_matched_pose_new(:,inlier_mask_tri>0);
        tri_p_W = tri_p_W(:,inlier_mask_tri>0);
    end
    
    % Print message - debug only
    if exist('inlier_mask_tri','var')
        fprintf('         Punti triangolati validi %d / %d su un totale %d\n\n', ...
            nnz(inlier_mask_tri),length(idx),size(tri_matched_new_h,2));
    else
        fprintf('         Punti triangolati validi %d su %d\n\n', nnz(idx),size(tri_matched_new_h,2));
    end
    
    % Plot - debug only
    if debug_plot
        figure(3);
        P = tri_p_W;
        plot3(P(1,:), P(2,:), P(3,:), 'o',p_W_landmarks(1,:),p_W_landmarks(2,:),p_W_landmarks(3,:),'*');
        grid on; xlabel('x'), ylabel('y'), zlabel('z');
        plotCoordinateFrame(eye(3),zeros(3,1), 0.8);
        text(-0.1,-0.1,-0.1,'Cam 1','fontsize',10,'color','k','FontWeight','bold');
        center_cam2_W = -R_C_W'*t_C_W;
        plotCoordinateFrame(R_C_W',center_cam2_W, 0.8);
        text(center_cam2_W(1)-0.1, center_cam2_W(2)-0.1, center_cam2_W(3)-0.1,'Cam 2','fontsize',10,'color','k','FontWeight','bold');
        rotate3d on; grid on;
    end
    
    % Creation of the new state
    new_state = [new_keypoints',p_W_landmarks',M_new';
        tri_matched_new', tri_p_W', tri_matched_pose_new';
        matched_old', NaN(size(matched_pose_old,2),3), matched_pose_old';
        not_matched_new', NaN(size(not_matched_new,2),3), not_matched_pose_new'];
else
    % Creation of the new state
    new_state = [new_keypoints',p_W_landmarks',M_new';
        matched_old', NaN(size(matched_pose_old,2),3), matched_pose_old';
        not_matched_new', NaN(size(not_matched_new,2),3), not_matched_pose_new'];
    fprintf('\n');
end



end


