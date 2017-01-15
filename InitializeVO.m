function [ state, W_T_c ] = InitializeVO( frame_1, frame_2, K )

% This function is set to initialize the VO pipeline. In particular, it is
% a monocular initialization.
% Inputs : - frame_1 = first image frame;
%          - frame_2 = second image frame;
%          - K       = intrinsic camera matrix;
% Outputs: - pose    = initial pose of the camera;
%          - state   = initial state.

% Initial parameters set
descriptor_radius = 9;
match_lambda = 8;

%% 1. Establish point correspondences
% 1.1 Extract Harris keypoints
% Keypoints frame 1
keypoints_1 = detectHarrisFeatures(frame_1,'MinQuality',0.1);
keypoints_2 = detectHarrisFeatures(frame_2,'MinQuality',0.1);
keypoints_1 = flipud(keypoints_1.Location');
keypoints_2 = flipud(keypoints_2.Location');
descriptors_1 = ourDescribeKeypoints(frame_1,keypoints_1,descriptor_radius);
descriptors_2 = ourDescribeKeypoints(frame_2,keypoints_2,descriptor_radius);

% 1.3 Match descriptors
matches = matchDescriptors(descriptors_2,descriptors_1,match_lambda);
% 1.4 Extract only matched points
keypoints_1 = keypoints_1(1:2,matches(matches>0));
keypoints_2_not_matched = detectHarrisFeatures(frame_2,'MinQuality',0.1);
keypoints_2_not_matched = double(keypoints_2_not_matched.Location');
keypoints_2 = keypoints_2(1:2,matches>0);

% Re-invert to have (u,v) coordinates
keypoints_1 = flipud(keypoints_1);
keypoints_2 = flipud(keypoints_2);

%% 2. Estimate the calibration matricies
% 2.1 Estimate the essential matrix E (here assume K1 = K2 = K since
%     camera 1 and camera 2 are the same cameras).
% Homogenous coordinates
key_h1 = [keypoints_1; ones(1,size(keypoints_1,2))];
key_h2 = [keypoints_2; ones(1,size(keypoints_2,2))];
% Perform 8-point Algorithm with RANSAC
[E, inlier_mask] = ourEstimateEssentialMatrix(key_h1,key_h2,K,K);

key_h1 = key_h1(:,inlier_mask>0);
key_h2 = key_h2(:,inlier_mask>0);

% 2.2 Obtain extrinsic parameters (R,t) from E
[Rots,t] = decomposeEssentialMatrix(E);

% 2.3 Disambiguate among the four possible configurations
[R_C2_W,T_C2_W] = disambiguateRelativePose(Rots,t,key_h1,key_h2,K,K);

% 2.4 Obtain the final pose
M1 = K * eye(3,4);              % Camera 1
M2 = K * [R_C2_W, T_C2_W];      % Camera 2

% 2.5 Triangulate a point cloud using the final transformation (R,T)
p_W_landmarks = linearTriangulation(key_h1,key_h2,M1,M2);
% Extract only sensible 3D points (ie: no negative depth). Moreover, 
% discard the 4th coordinate (it is always 1).
idx = p_W_landmarks(3,:)>0;
p_W_landmarks = p_W_landmarks(1:3,idx);
key_h2 = key_h2(:,idx);

%% Function outputs
% The state repeates the 2D keypoints for the new tringulation in
% processFrame.m. 
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
    keypoints_2_not_matched', keypoints_2_not_matched', NaN(size(keypoints_2_not_matched,2),1), M_not_matched'];

fprintf('The intial posepose is:\n');
disp(W_T_c);
fprintf('\nThe number of valid correspondences in the initialization is %d.',find(isnan(state(:,3)),1,'first')-1);

end

