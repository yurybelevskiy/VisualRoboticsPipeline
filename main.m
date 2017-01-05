%% Setup

clear all;
close all;
clc;

ds = 0; % 0: KITTI, 1: Malaga, 2: parking

if ds == 0
    % need to set kitti_path to folder containing "00" and "poses"
    kitti_path = [pwd, '\kitti'];
    assert(exist('kitti_path', 'var') ~= 0);
    ground_truth = load([kitti_path '/poses/00.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
elseif ds == 1
    % Path containing the many files of Malaga 7.
    malaga_path = [pwd, '\malaga'];
    assert(exist('malaga_path', 'var') ~= 0);
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif ds == 2
    % Path containing images, depths and all...
    parking_path = [pwd, '\parking'];
    assert(exist('parking_path', 'var') ~= 0);
    last_frame = 598;
    K = load([parking_path '/K.txt']);
     
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
else
    assert(false);
end

% Stereo initialization
frame_1 = imread([kitti_path '/00/image_0/000000.png']);
% frame_2 = imread([kitti_path '/00/image_1/000000.png']); % stereo
frame_2 = imread([kitti_path '/00/image_0/000002.png']); % mono
[initial_state, pose_init] = InitializeVO(frame_1,frame_2,K);
old_state = initial_state;
old_pose = [pose_init; 0,0,0,1]; % goes from CF3 to CF1 -> invert it!
old_pose = inv(old_pose);
pose_init = old_pose;

movement = zeros(1,2);
P_old = zeros(3,1);%old_pose(1:3,4);
concatenated = old_pose;
for k = 3:250
    fprintf('Step %d - ',k);
    previous_image = imread(sprintf('%s/00/image_0/%06d.png',kitti_path,k-1));
    new_image = imread(sprintf('%s/00/image_0/%06d.png',kitti_path,k));
    [new_state,new_pose] = processFrame(old_state,previous_image,new_image,K);
    old_state = new_state;
    new_pose = inv([new_pose; 0,0,0,1]);
    new_pose_m = new_pose;
%     [P_new, concatenated] = computeMovement(new_pose,old_pose,P_old);
    concatenated = concatenated*new_pose;
%     P_old = P_new;
%     old_pose = concatenated;
%     movement(k-3,:) = [P_new(1), P_new(3)]; 
    movement(k-2,:) = [concatenated(1,4), concatenated(3,4)];
    m(k-2,:) = [new_pose_m(1,end), new_pose_m(3,end)];
    total_pose(k-2,:) = new_pose(1:3,4);
    
%     figure(10);
%     plot(ground_truth(k,1),ground_truth(k,2),'b-o',m(k-2,1),m(k-2,2),'r-*');
%     hold on;    
%     pause(0.1);
end

n = size(total_pose,1);
% p_W_estimate_aligned = alignEstimateToGroundTruth(ground_truth(1:n,:)', total_pose');


figure
plot(movement(:,1),movement(:,2),'-*');
%% Bootstrap
% need to set bootstrap_frames
if ds == 0
    img0 = imread([kitti_path '/00/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(1))]);
    img1 = imread([kitti_path '/00/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(2))]);
elseif ds == 1
    img0 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(1)).name]));
    img1 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(2)).name]));
elseif ds == 2
    img0 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',bootstrap_frames(1))]));
    img1 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',bootstrap_frames(2))]));
else
    assert(false);
end



%% Continuous operation
range = (bootstrap_frames(2)+1):last_frame;
for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    if ds == 0
        image = imread([kitti_path '/00/image_0/' sprintf('%06d.png',i)]);
    elseif ds == 1
        image = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
    elseif ds == 2
        image = im2uint8(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',i)])));
    else
        assert(false);
    end
    % Makes sure that plots refresh.    
    pause(0.01);
    
    prev_img = image;
end