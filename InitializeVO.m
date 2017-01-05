function [ state, W_T_c ] = InitializeVO( frame_1, frame_2, K )

% This function is set to initialize the VO pipeline. In particular, it is
% possible to choose between stereo initialization and monocular
% initialization.
% Inputs : - frame_1 = first image frame (left in stereo case);
%          - frame_2 = second image frame (right in stereo case);
%          - K       = intrinsic camera matrix;
% Outputs: - pose    = initial pose of the camera;
%          - state   = initial state.

debug_plot = 0;
debug_plot_3d = 0;

% Initial parameters set
harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 1000;
nonmaximum_supression_radius = 6;
descriptor_radius = 9;
match_lambda = 8;

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

keypoints_1 = detectHarrisFeatures(frame_1);
keypoints_2 = detectHarrisFeatures(frame_2);
keypoints_1 = flipud(keypoints_1.Location');
keypoints_2 = flipud(keypoints_2.Location');
descriptors_1 = ourDescribeKeypoints(frame_1,keypoints_1,descriptor_radius);
descriptors_2 = ourDescribeKeypoints(frame_2,keypoints_2,descriptor_radius);

% 1.3 Match descriptors
% PROBLEM HERE IF WE USE matchHarrisFeature!!!
matches = matchDescriptors(descriptors_2,descriptors_1,match_lambda);
% 1.4 Extract only matched points
keypoints_1 = keypoints_1(1:2,matches(matches>0));
keypoints_2_not_matched = flipud(keypoints_2(1:2,matches==0));
keypoints_2 = keypoints_2(1:2,matches>0);

% Invert order of rows to be consistent with the function of exe5
keypoints_1 = flipud(keypoints_1);
keypoints_2 = flipud(keypoints_2);
% Plot of the matched features
if debug_plot
    % Extract keypoints to be plotted (debugging)
    p1_plot = keypoints_1;
    p2_plot = keypoints_2;
    plotMatched(frame_1,p1_plot,p2_plot,false,6);
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
    figure(1), imshow(frame_1), hold on, plot(keypoints_1(1,:),keypoints_1(2,:),'rx','linewidth',2); hold off;
    figure(2), imshow(frame_2), hold on, plot(keypoints_2(1,:),keypoints_2(2,:),'rx','linewidth',2); plot(keypoints_2_not_matched(1,:),keypoints_2_not_matched(2,:),'gx','linewidth',2); hold off;
    plotMatched(frame_1,p1_plot,p2_plot,true,3);
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
idx = p_W_landmarks(3,:)>0 & p_W_landmarks(3,:)<75;
p_W_landmarks = p_W_landmarks(1:3,idx);
key_h2 = key_h2(:,idx);

% Plot the 3d points
if debug_plot_3d
    figure(4);
    P = p_W_landmarks;
    plot3(P(1,:), P(2,:), P(3,:), 'o'); grid on;
    xlabel('x'), ylabel('y'), zlabel('z');
    plotCoordinateFrame(eye(3),zeros(3,1), 0.8);
    text(-0.1,-0.1,-0.1,'Cam 1','fontsize',10,'color','k','FontWeight','bold');
    center_cam2_W = -R_C2_W'*T_C2_W;
    plotCoordinateFrame(R_C2_W',center_cam2_W, 0.8);
%     text(center_cam2_W(1)-0.1, center_cam2_W(2)-0.1, center_cam2_W(3)-0.1,'Cam 2','fontsize',10,'color','k','FontWeight','bold');
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
M_not_matched = repmat(reshape(W_T_c,12,1),1,size(keypoints_2_not_matched,2));
% STATE:
% - keypoints in the initialization
% - 3D corresponding points
% - initialization of keypoints tracking for triangulation
% - Camera pose for each point
state = [key_h2(1:2,:)', p_W_landmarks', M';
    keypoints_2_not_matched', NaN(size(keypoints_2_not_matched,2),3), M_not_matched'];

% Debug only: display results
fprintf('The pose is: [%f %f %f %f\n        %f %f %f %f\n        %f %f %f %f]\n',W_T_c);
fprintf('The number of valid correspondences is %d\n',find(isnan(state(:,3:5)),1,'first')-1);

end

