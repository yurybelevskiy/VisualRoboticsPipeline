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

% Initial parameters set
harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 800;
nonmaximum_supression_radius = 6;
descriptor_radius = 9;
match_lambda = 8; % <---------- TO BE TUNED PROPERLY FOR THE DESCRIPTOR MATCHING!!!!

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
    keypoints_1 = detectHarrisFeatures(frame_1);
    keypoints_1 = keypoints_1.Location';
    keypoints_2 = detectHarrisFeatures(frame_2);
    keypoints_2 = keypoints_2.Location';
    descriptors_1 = ourDescribeKeypoints(frame_1,keypoints_1,descriptor_radius);
    descriptors_2 = ourDescribeKeypoints(frame_2,keypoints_2,descriptor_radius);
%     % TO CHOOSE BETWEEN WHICH FUNCTIONS TO USE
%     scores_1 = harris(frame_1,harris_patch_size,harris_kappa);
%     keypoints_1 = selectKeypoints(...
%         scores_1, num_keypoints, nonmaximum_supression_radius);
%     scores_2 = harris(frame_2,harris_patch_size,harris_kappa);
%     keypoints_2 = selectKeypoints(...
%         scores_2, num_keypoints, nonmaximum_supression_radius);
%     % 1.2 Extract descriptors corresponding to keypoints
%     descriptors_1 = describeKeypoints(frame_1,keypoints_1,descriptor_radius);
%     descriptors_2 = describeKeypoints(frame_2,keypoints_2,descriptor_radius);

    % 1.3 Match descriptors 
    % PROBLEM HERE IF WE USE matchHarrisFeature!!!
    matches = matchDescriptors(descriptors_2,descriptors_1,match_lambda); 
    % 1.4 Extract only matched points
    keypoints_1 = keypoints_1(1:2,matches(matches>0));
    keypoints_2 = keypoints_2(1:2,matches>0);
    
    % IMAGE TO DEBUG RANSAC
    % Plot the matched features
%     figure(1);
%     keypoints_1 = [keypoints_1, [1;1], [2;2], [3;3]]; keypoints_2 = [keypoints_2, [376;376], [1239;1240], [1238;1240]];
%     plot_point1 = [keypoints_1(2,:)',keypoints_1(1,:)'];
%     plot_point2 = [keypoints_2(2,:)',keypoints_2(1,:)'];
%     showMatchedFeatures(frame_1,frame_2,plot_point1,plot_point2,'montage','PlotOptions',{'ro','go','y--'});
%     title('Without RANSAC');
        
    % 2. Estimate the calibration matricies
    % 2.1 Estimate the essential matrix E (here assume K1 = K2 = K since
    %     camera 1 and camera 2 are the same cameras).
    % Homogenous coordinates
    homo_points_1 = transformIntoHomogenous(keypoints_1); %normc(transformIntoHomogenous(keypoints_1));
    homo_points_2 = transformIntoHomogenous(keypoints_2); %normc(transformIntoHomogenous(keypoints_2));
    % RANSAC TO BE IMPROVED : now it works at least!!
    [E, inlier_mask] = ourEstimateEssentialMatrix(homo_points_1,homo_points_2,K,K); 
    
    keypoints_1_inlier = keypoints_1(1:2,inlier_mask>0);
    keypoints_2_inlier = keypoints_2(1:2,inlier_mask>0);
%     % Plot the matching 
%     figure(2);
%     plot_point1 = [keypoints_1_inlier(2,:)',keypoints_1_inlier(1,:)'];
%     plot_point2 = [keypoints_2_inlier(2,:)',keypoints_2_inlier(1,:)'];
%     showMatchedFeatures(frame_1,frame_2,plot_point1,plot_point2,'montage','PlotOptions',{'ro','go','--'});
%     title('With RANSAC');
    
    % 2.2 Obtain extrinsic parameters (R,t) from E
    [Rots,t] = decomposeEssentialMatrix(E);
    
    % 2.3 Disambiguate among the four possible configurations
    [R_C2_W,T_C2_W] = disambiguateRelativePose(Rots,t,homo_points_1,homo_points_2,K,K);
    
    % 2.4 Obtain the final pose
    M1 = K * eye(3,4);             % Camera 1
    M2 = K * [R_C2_W, T_C2_W];      % Camera 2
    
    % PROBLEM HERE: the z of the 3d points makes no sense! --> TO DO
    % 2.5 Triangulate a point cloud using the final transformation (R,T)
    p_W_landmarks = linearTriangulation(homo_points_1,homo_points_2,M1,M2);
    p_W_landmarks = p_W_landmarks(1:3,inlier_mask>0);
    
    % Function outputs
    % The state repeates the 2d keypoints for the new tringulation in
    % processFrame.m. Moreover, we add the matrix M 4x4 as a 16x1 vector.
    W_T_c = [R_C2_W, T_C2_W];
    % Create the 4x4 matrix and shape it in a 16x1 vector. Repeat it for
    % every keypoint.
    M = repmat(reshape([W_T_c; 0,0,0,1],16,1),1,size(keypoints_2_inlier,2));
    state = [keypoints_2_inlier; p_W_landmarks; keypoints_2_inlier; M];
  
    %% DEBUGGING
% % % %     Parallel with matlab function
% % %     keypoints_1_m = detectHarrisFeatures(frame_1);
% % %     keypoints_1_m = keypoints_1_m.Location;
% % %     keypoints_2_m = detectHarrisFeatures(frame_2);
% % %     keypoints_2_m = keypoints_2_m.Location;
% % %     
% % %     [features1,valid_points1] = extractFeatures(frame_1,keypoints_1_m);
% % %     [features2,valid_points2] = extractFeatures(frame_2,keypoints_2_m);
% % %     indexPairs = matchFeatures(features1,features2);
% % %     matchedPoints1 = valid_points1(indexPairs(:,1),:);
% % %     matchedPoints2 = valid_points2(indexPairs(:,2),:);
% % %     F_matlab = estimateFundamentalMatrix(matchedPoints1,matchedPoints2)

% % %     
% % %     p1 = transformIntoHomogenous(keypoints_1); p1 = [p1, [1;1;1], [2;2;1], [3;3;1]];
% % %     p2 = transformIntoHomogenous(keypoints_2);  p2 = [p2, [1240;1240;1], [1239;1240;1], [1238;1240;1]];
% % %     [F_pk, inliers] = ransacfitfundmatrix(p1,p2,1);
% % %     figure(3);
% % %     plot_point1 = [p1(2,:)',p1(1,:)'];
% % %     plot_point2 = [p2(2,:)',p2(1,:)'];
% % %     showMatchedFeatures(frame_1,frame_2,plot_point1,plot_point2,'montage','PlotOptions',{'ro','go','y--'});hold on;
% % %     p1 = p1(1:2,inliers); p2 = p2(1:2,inliers);
% % %     plot_point1 = [p1(2,:)',p1(1,:)'];
% % %     plot_point2 = [p2(2,:)',p2(1,:)'];
% % %     showMatchedFeatures(frame_1,frame_2,plot_point1,plot_point2,'montage','PlotOptions',{'ro','go','c--'});
% % %     title('With RANSAC');
% % %     F_pk
end

end

