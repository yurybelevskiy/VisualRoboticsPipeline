%% SCRIPT TO TEST THE MONOCULAR INITIALIZATION

clear all;
close all;
clc;

debug = 0;
true_points = 0;
matlab = 0;
correct_matlab = 0;

% Path
kitti_path = [pwd, '\kitti'];
% Select frames
if ~debug
    frame_1 = imread([kitti_path '/00/image_0/000001.png']);
    frame_2 = imread([kitti_path '/00/image_0/000003.png']);
else
    frame_1 = double(rgb2gray(imread([pwd '/Debug/0001.png'])));
    frame_2 = double(rgb2gray(imread([pwd '/Debug/0002.png'])));
    frame_1_i = imread([pwd '/Debug/0001.png']);
    frame_2_i = imread([pwd '/Debug/0002.png']);
end

% Intrinsic matrix
if ~debug
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
else
    K = [1379.74 0 760.35
    0 1382.08 503.41
    0 0 1 ];
end
    
%% Establish keypoints correpondences

harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 300;
nonmaximum_supression_radius = 6; % 6
descriptor_radius = 9; % 9
match_lambda = 6;

if ~matlab && ~correct_matlab
    % Key 1
    scores_1 = harris(frame_1,harris_patch_size,harris_kappa);
    keypoints_1 = selectKeypoints(scores_1, num_keypoints, nonmaximum_supression_radius);
    % Key 2
    scores_2 = harris(frame_2,harris_patch_size,harris_kappa);
    keypoints_2 = selectKeypoints(scores_2, num_keypoints, nonmaximum_supression_radius);
    % Descriptor 1
    descriptors_1 = describeKeypoints(frame_1,keypoints_1,descriptor_radius);
    % Descriptor 1
    descriptors_2 = describeKeypoints(frame_2,keypoints_2,descriptor_radius);
    % Matches
    matches = matchDescriptors(descriptors_2,descriptors_1,match_lambda);
    keypoints_1 = keypoints_1(1:2,matches(matches>0));
    keypoints_2 = keypoints_2(1:2,matches>0);
    
    keypoints_1 = [keypoints_1(2,:); keypoints_1(1,:)];
    keypoints_2 = [keypoints_2(2,:); keypoints_2(1,:)];
    p1_plot = [keypoints_1(2,:); keypoints_1(1,:)];
    p2_plot = [keypoints_2(2,:); keypoints_2(1,:)];

elseif matlab
    keypoints_1 = detectHarrisFeatures(frame_1);
    keypoints_1 = keypoints_1.Location;
    keypoints_2 = detectHarrisFeatures(frame_2);
    keypoints_2 = keypoints_2.Location;
    
    [features1,valid_points1] = extractFeatures(frame_1,keypoints_1);
    [features2,valid_points2] = extractFeatures(frame_2,keypoints_2);
    indexPairs = matchFeatures(features1,features2);
    matchedPoints1 = valid_points1(indexPairs(:,1),:);
    matchedPoints2 = valid_points2(indexPairs(:,2),:);
    
elseif correct_matlab
    keypoints_1 = detectHarrisFeatures(frame_1);
    keypoints_1 = keypoints_1.Location';
    keypoints_2 = detectHarrisFeatures(frame_2);
    keypoints_2 = keypoints_2.Location';
    keypoints_1 = [keypoints_1(2,:); keypoints_1(1,:)];
    keypoints_2 = [keypoints_2(2,:); keypoints_2(1,:)];
    descriptors_1 = ourDescribeKeypoints(frame_1,keypoints_1,descriptor_radius);
    descriptors_2 = ourDescribeKeypoints(frame_2,keypoints_2,descriptor_radius);
    matches = matchDescriptors(descriptors_2,descriptors_1,match_lambda);
    keypoints_1 = keypoints_1(1:2,matches(matches>0));
    keypoints_2 = keypoints_2(1:2,matches>0);
    keypoints_1 = [keypoints_1(2,:); keypoints_1(1,:)];
    keypoints_2 = [keypoints_2(2,:); keypoints_2(1,:)];
    p1_plot = [keypoints_1(2,:); keypoints_1(1,:)];
    p2_plot = [keypoints_2(2,:); keypoints_2(1,:)];

end
    
    
% Plot -> correct matches? (WITHOUT ransac)
if ~debug
    plotMatched(frame_1,p1_plot,p2_plot,false);
end

% %% Estimate relative pose between frames
if debug && true_points
    keypoints_1 = load('matches0001_debug.txt');
    keypoints_2 = load('matches0002_debug.txt');
%     keypoints_1 = [keypoints_1(2,:); keypoints_1(1,:)];
%     keypoints_2 = [keypoints_2(2,:); keypoints_2(1,:)];
    plotMatched(frame_1_i,keypoints_1,keypoints_2,false);
end

% Homogeneous coordinates
if ~matlab
%     keypoints_1 = [keypoints_1(2,:); keypoints_1(1,:)];
%     keypoints_2 = [keypoints_2(2,:); keypoints_2(1,:)];
    key_h1 = [keypoints_1; ones(1,size(keypoints_1,2))];
    key_h2 = [keypoints_2; ones(1,size(keypoints_2,2))];
    [E, inlier_mask] = ourEstimateEssentialMatrix(key_h1,key_h2,K,K);
%     E = estimateEssentialMatrix(key_h1,key_h2,K,K), inlier_mask = ones(1,size(keypoints_1,2));
    % Extract only the inliersfalse
    keypoints_1 = keypoints_1(:,inlier_mask>0);
    keypoints_2 = keypoints_2(:,inlier_mask>0);
    key_h1 = key_h1(:,inlier_mask>0);
    key_h2 = key_h2(:,inlier_mask>0);
    p1_plot = [keypoints_1(2,:); keypoints_1(1,:)];
    p2_plot = [keypoints_2(2,:); keypoints_2(1,:)];
else
    [F_matlab, inlier_mask] = estimateFundamentalMatrix(matchedPoints1,matchedPoints2);
    E = K'*F_matlab*K
    % Extract only the inliers
    keypoints_1 = [matchedPoints1(:,2), matchedPoints1(:,1)];
    keypoints_2 = [matchedPoints2(:,2), matchedPoints2(:,1)];
    keypoints_1 = keypoints_1(inlier_mask>0,:)';
    keypoints_2 = keypoints_2(inlier_mask>0,:)';
    key_h1 = [keypoints_1; ones(1,size(keypoints_1,2))];
    key_h2 = [keypoints_2; ones(1,size(keypoints_2,2))];
end


% Plot (WITH ransac)
if ~debug
    plotMatched(frame_1,p1_plot,p2_plot,true);
end

[Rots,t] = decomposeEssentialMatrix(E);
[R_C2_W,T_C2_W] = disambiguateRelativePose(Rots,t,key_h1,key_h2,K,K);

%% Triangulation 3D positions
M1 = K * eye(3,4);              % Camera 1
M2 = K * [R_C2_W, T_C2_W];      % Camera 2

p_W_landmarks = linearTriangulation(key_h1,key_h2,M1,M2);

idx = find(p_W_landmarks(3,:)>0 & p_W_landmarks(3,:)<50);

% Plot correpondece
figure(3);
subplot(211); imshow(frame_1); hold on; plot(p1_plot(2,:),p1_plot(1,:),'rx','linewidth',1.25); hold off;
subplot(212); imshow(frame_2); hold on; plot(p2_plot(2,:),p2_plot(1,:),'go','linewidth',1.25); hold off;

% Plot the 3d points
figure(4);
P = p_W_landmarks(:,idx);
plot3(P(1,:), P(2,:), P(3,:), 'o');

% Display camera pose
plotCoordinateFrame(eye(3),zeros(3,1), 0.8);
text(-0.1,-0.1,-0.1,'Cam 1','fontsize',10,'color','k','FontWeight','bold');
center_cam2_W = -R_C2_W'*T_C2_W;
plotCoordinateFrame(R_C2_W',center_cam2_W, 0.8);
text(double(center_cam2_W(1))-0.1, double(center_cam2_W(2))-0.1, double(center_cam2_W(3))-0.1,'Cam 2','fontsize',10,'color','k','FontWeight','bold');
xlabel('x'), ylabel('y'), zlabel('z');
% axis equal
rotate3d on;
grid


%% Plot correct 3d points
load('p_W_landmarks.txt');
% figure(5);
% plot3(p_W_landmarks(:,1), p_W_landmarks(:,2), p_W_landmarks(:,3), 'o'); grid on;
% xlabel('x'), ylabel('y'), zlabel('z');
% rotate3d on;



