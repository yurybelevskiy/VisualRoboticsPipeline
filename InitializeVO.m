function [ state, W_T_c ] = InitializeVO( frame_1, frame_2, K )

% This function is set to initialize the VO pipeline. In particular, it is
% possible to choose between stereo initialization and monocular
% initialization.
% Inputs : - frame_1 = first image frame (left in stereo case);
%          - frame_2 = second image frame (right in stereo case);
%          - K       = intrinsic camera matrix;
% Outputs: - pose    = initial pose of the camera;
%          - state   = initial state.

use_stereo = false;
debug_plot = false;

% Initial parameters set
harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 300; % <---------- NOT TOO MANY MATCHES, BUT DECENT RESULTS
nonmaximum_supression_radius = 6;
descriptor_radius = 9;
match_lambda = 6; % <---------- TO BE TUNED PROPERLY FOR THE DESCRIPTOR MATCHING!!!!

patch_radius = 5;
min_disp = 5;
max_disp = 50;

% For KITTY dataset 
baseline = 0.54;

if use_stereo
    img_left = frame_1;
    img_right = frame_2;
    % Extraction of keypoints set in I0_left (Harris detector)
%     scores_left = harris(img_left,harris_patch_size,harris_kappa);
%     assert(min(size(scores_left) == size(img_left))); % Check-point
    % Select the best keypoints in the left image
%     keypoints_left = selectKeypoints(...
%         scores_left, num_keypoints, nonmaximum_supression_radius);
    keypoints_left = detectHarrisFeatures(frame_1);
    keypoints_left = keypoints_left.Location';
    
    % Establish correspondences
    disp_img = ourGetDisparity(img_left,img_right,patch_radius,min_disp,max_disp);
    % 3D landmarks points (create right correspondence 2D-3D points)
    [p_W_landmarks, keypoints_left] = ourDisparityToPointCloud(...
        disp_img,K,baseline,img_left,keypoints_left);
    % Function outputs
    state = [keypoints_left; p_W_landmarks];
    W_T_c = K; % TO BE SET
else
    % frame_1 = first of the image pairs; for kitti dataset : frame 1;
    % frame_2 = second of the image pairs; for kitti dataset : frame 3.
    
    % 1. Establish point correspondences
    % 1.1 Extract Harris keypoints
    % Keypoints frame 1
    scores_1 = harris(frame_1,harris_patch_size,harris_kappa);
    keypoints_1 = selectKeypoints(...
        scores_1, num_keypoints, nonmaximum_supression_radius);
    % Keypoints frame 2
    scores_2 = harris(frame_2,harris_patch_size,harris_kappa);
    keypoints_2 = selectKeypoints(...
        scores_2, num_keypoints, nonmaximum_supression_radius);
    % 1.2 Extract descriptors corresponding to keypoints
    descriptors_1 = describeKeypoints(frame_1,keypoints_1,descriptor_radius);
    descriptors_2 = describeKeypoints(frame_2,keypoints_2,descriptor_radius);

    % 1.3 Match descriptors 
    % PROBLEM HERE IF WE USE matchHarrisFeature!!!
    matches = matchDescriptors(descriptors_2,descriptors_1,match_lambda); 
    % 1.4 Extract only matched points
    keypoints_1 = keypoints_1(1:2,matches(matches>0));
    keypoints_2 = keypoints_2(1:2,matches>0);
    % Extract keypoints to be plotted (debugging)
    p1_plot = keypoints_1;
    p2_plot = keypoints_2;
    % Invert order of rows to be consistent with the function of exe5
    keypoints_1 = flipud(keypoints_1);
    keypoints_2 = flipud(keypoints_2);
    % Plot of the matched features
    if debug_plot
        plotMatched(frame_1,p1_plot,p2_plot,false);
    end
        
    % 2. Estimate the calibration matricies
    % 2.1 Estimate the essential matrix E (here assume K1 = K2 = K since
    %     camera 1 and camera 2 are the same cameras).
    % Homogenous coordinates
    key_h1 = [keypoints_1; ones(1,size(keypoints_1,2))];
    key_h2 = [keypoints_2; ones(1,size(keypoints_2,2))];
    % Perform 8-point Algorithm with RANSAC
    [E, inlier_mask] = ourEstimateEssentialMatrix(key_h1,key_h2,K,K);
    key_h1 = key_h1(:,inlier_mask>0);
    key_h2 = key_h2(:,inlier_mask>0);
    % Extract keypoints to be plotted (debugging)
    p1_plot = [key_h1(2,:); key_h1(1,:)];
    p2_plot = [key_h2(2,:); key_h2(1,:)];
    % Plot of the matched features
    if debug_plot
        plotMatched(frame_1,p1_plot,p2_plot,true);
    end
    
    % 2.2 Obtain extrinsic parameters (R,t) from E
    [Rots,t] = decomposeEssentialMatrix(E);
    
    % 2.3 Disambiguate among the four possible configurations
    [R_C2_W,T_C2_W] = disambiguateRelativePose(Rots,t,key_h1,key_h2,K,K);
    
    % 2.4 Obtain the final pose
    M1 = K * eye(3,4);              % Camera 1
    M2 = K * [R_C2_W, T_C2_W];      % Camera 2
    
    % There have been several problems here: check it out! If something
    % does not convince you, just let me know!
    % 2.5 Triangulate a point cloud using the final transformation (R,T)
    p_W_landmarks = linearTriangulation(key_h1,key_h2,M1,M2);
    % Extract only sensible 3D points (ie: no negative depth; no farther
    % than 50 meter from our current position). Moreover, discard the 4th
    % coordinate (it is always 1: they are homogeneous coordinates!)
    idx = p_W_landmarks(3,:)>0 & p_W_landmarks(3,:)<50;
    p_W_landmarks = p_W_landmarks(1:3,idx);
    key_h2 = key_h2(:,idx);
          
    % Plot the 3d points
    if debug_plot
        figure(3);
        P = p_W_landmarks;
        plot3(P(1,:), P(2,:), P(3,:), 'o'); grid on;
        xlabel('x'), ylabel('y'), zlabel('z');
        plotCoordinateFrame(eye(3),zeros(3,1), 0.8);
        text(-0.1,-0.1,-0.1,'Cam 1','fontsize',10,'color','k','FontWeight','bold');
        center_cam2_W = -R_C2_W'*T_C2_W;
        plotCoordinateFrame(R_C2_W',center_cam2_W, 0.8);
        text(center_cam2_W(1)-0.1, center_cam2_W(2)-0.1, center_cam2_W(3)-0.1,'Cam 2','fontsize',10,'color','k','FontWeight','bold');
        rotate3d on;
        grid on;
    end
    
    % Function outputs
    % The state repeates the 2D keypoints for the new tringulation in
    % processFrame.m. Moreover, we add the matrix M 4x4 as a 16x1 vector.
    W_T_c = [R_C2_W, T_C2_W];
    % Create the 4x4 matrix and shape it in a 16x1 vector. Repeat it for
    % every keypoint.
    M = repmat(reshape(W_T_c,12,1),1,size(key_h2,2));
    M_not_matched = repmat(reshape(W_T_c,12,1),1,size(keypoints_2,2));
    % STATE:
    % - keypoints in the initialization
    % - 3D corresponding points
    % - initialization of keypoints tracking for triangulation
    % - Camera pose for each point
%     state = [key_h2(1:2,:); p_W_landmarks; key_h2(1:2,:); M];
    state = [[key_h2(1:2,:)', p_W_landmarks', M']; 
        keypoints_2', NaN(size(keypoints_2,2),3), M_not_matched'];
%         key_h2(1:2,:)', -ones(size(key_h2,2),3), M'];
  
end
end
