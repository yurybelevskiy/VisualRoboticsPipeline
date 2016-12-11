function [ state ] = InitializeVO( frame_1, frame_2, K )

% This function is set to initialize the VO pipeline. In particular, it is
% possible to choose between stereo initialization and monocular
% initialization.
% Inputs : - frame_1 = first image frame (left in stereo case);
%          - frame_2 = second image frame (right in stereo case);
%          - K       = intrinsic camera matrix;
% Outputs: - pose    = initial pose of the camera;
%          - state   = initial state.

use_stereo = true;

% Initial parameters set
harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 516;
nonmaximum_supression_radius = 6;
patch_radius = 5;
min_disp = 5;
max_disp = 50;

% For KITTY dataset 
baseline = 0.54;

if use_stereo
    img_left = frame_1;
    img_right = frame_2;
    % Extraction of keypoints set in I0_left (Harris detector)
    scores_left = harris(img_left,harris_patch_size,harris_kappa);
    assert(min(size(scores_left) == size(img_left))); % Check-point
    % Select the best keypoints in the left image
    keypoints_left = selectKeypoints(...
        scores_left, num_keypoints, nonmaximum_supression_radius);
% %     keypoints_left = detectHarrisFeatures(frame_1);
% %     keypoints_left = keypoints_left.Location';
    
    % Establish correspondences
    disp_img = 0;%ourGetDisparity(img_left,img_right,patch_radius,min_disp,max_disp);
    % 3D landmarks points (create right correspondence 2D-3D points)
    [p_W_landmarks, keypoints_left] = ourDisparityToPointCloud(...
        disp_img,K,baseline,img_left,keypoints_left);
    
   % Function outputs
    state = [keypoints_left; p_W_landmarks];
end

% MONOCULAR : to be done
end

