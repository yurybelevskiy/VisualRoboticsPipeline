%% Setup

clear all;
close all;
clc;

ds = 2; % 0: KITTI, 1: Malaga, 2: parking

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

%% Bootstrap
if ds == 0 || ds == 1
    bootstrap_frames = [1, 3];
elseif ds == 2
    bootstrap_frames = [0,2];
end

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

[initial_state, pose_init] = InitializeVO(img0,img1,K);
old_state = initial_state;
old_pose = [pose_init; 0,0,0,1]; % goes from CF3 to CF1 -> invert it!
old_pose = inv(old_pose);
pose_init = old_pose;


%% Continuous operation
range = (bootstrap_frames(2)+1):last_frame;
prev_img = img1;
number_valid_landmarks = zeros(length(range),1);
current_position = zeros(length(range),2);
figure(1);
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
       
    [new_state,new_pose,error] = processFrame(old_state,prev_img,image,K,i);
    if error < 0
        fprintf('********************************************************************\n');
        fprintf('ERROR!\nThe process has been interrupted at frame number %d since there are not any more valid points.\n',i-bootstrap_frames(2));
        fprintf('********************************************************************\n');
        break;
    end
    
    % The result from P3P.m goes from the camera to the world, so it must
    % be inverted.
    new_pose_m = inv([new_pose; 0,0,0,1]);
    current_position(i-bootstrap_frames(2),:) = [new_pose_m(1,end), new_pose_m(3,end)];
    
    %% Plot
    % Create variables for the plot of the results
    stop_old = find(isnan(old_state(:,5)),1,'first')-1;
    stop_new = find(isnan(new_state(:,5)),1,'first')-1;
    
    old_keypoints = old_state(1:stop_old,1:2)';     % Keypoints in previous state
    p_W_landmarks_old = old_state(1:stop_old,3:5)'; % Landmarks in previous state
    p_W_landmarks_new = new_state(1:stop_new,3:5)'; % Landmarks in new state
    if ~isempty(stop_new)
        new_keypoints = new_state(1:stop_new,1:2)'; % Keypoints in the new state
    else
        new_keypoints = new_state(1:end,1:2)';      % Keypoints in the new state
    end
    
    % Plot of the keypoints in the current frame
    subplot(2,3,[1 2]);
    imshow(image); hold on; 
    title('Current frame');
    plot(old_keypoints(1,:),old_keypoints(2,:),'gx','linewidth',1.25);
    plot(new_keypoints(1,:),new_keypoints(2,:),'rx','linewidth',1.25);
    hold off;
    
    % Plot of the trajectory over last 20 frames with the last triangulated
    % points.
    if i-bootstrap_frames(2) > 20
        position_plot = current_position(i-bootstrap_frames(2)-20:i-bootstrap_frames(2),:);
    else
        position_plot = current_position(1:i-bootstrap_frames(2),:);
    end
    % Select only the landmarks in a certain range to make plot nicer.
    idx = p_W_landmarks_new(3,:) < 150 & p_W_landmarks_new(1,:)<200;
    p_W_plot = p_W_landmarks_new(:,idx);
    
    subplot(2,3,[3 6]);
    plot(current_position(:,1),current_position(:,2),'bo');
    
    hold on; 
    scatter(p_W_plot(1,:),p_W_plot(3,:),'MarkerEdgeColor',[0 .5 .5],'MarkerFaceColor',[0 .4 .4]); 
    hold off;
    % Adjust plot settings
    title('Camera position - (x,z) plane');
    grid on; 
    dummy_x = abs(mean(p_W_plot(1,:))); 
    dummy_y = abs(mean(p_W_plot(3,:)));
    xlim([current_position(i-bootstrap_frames(2),1)-dummy_x,...
        current_position(i-bootstrap_frames(2),1)+dummy_x]);
    ylim([current_position(i-bootstrap_frames(2),2)-dummy_y,...
        current_position(i-bootstrap_frames(2),2)+dummy_y]);
    xlabel('x [m]'); ylabel('z [m]');
    
    % Plot of size of number of keypoints triangulated.
    % Save the number of valid 3D points
    number_valid_landmarks(i-bootstrap_frames(2)) = size(p_W_landmarks_new,2);
    subplot(2,3,[4 5]);
    plot(1:i-bootstrap_frames(2),number_valid_landmarks(1:i-bootstrap_frames(2)));
    hold off;
    axis tight;
    grid on;
    xlabel('Number of frame'); ylabel('# of valid landmarks');
    title('# of valid landmarks - overall frames');
    pause(0.01);
    
    % Update variables
    old_state = new_state;
    prev_img = image;
end