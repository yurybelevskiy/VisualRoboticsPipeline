function [R_C_W, t_C_W, inlier_mask] = ourRansacLocalization(...
    query_keypoints, p_W_landmarks, K, pixel_tolerance)
% The variables query_keypoints and p_W_landmarks have already been
% filtered trough the Lukas-Kanade Tracker. Thus, there is no need for
% another matching operation.
% inlier_mask should be 1xnum_matched and contain, only for the
%   matched keypoints, 0 if the match is an outlier, 1 otherwise.

% Parameters initialization.
p = 0.99;                   % Probability that a sample is free from outliers
e = 0.55;                   % Probability that a point is an outlier
s = 3;                      % Number of point of correspondence
num_iterations = log(1-p)/log(1-(1-e).^s);

% Initialize RANSAC.
inlier_mask = zeros(1, size(query_keypoints, 2));
max_num_inliers_history = zeros(1, ceil(num_iterations));
max_num_inliers = 0;

% RANSAC iterations.
for i = 1:num_iterations
    [landmark_sample, idx] = datasample(...
        p_W_landmarks, s, 2, 'Replace', false);
    keypoint_sample = query_keypoints(:, idx);
    
    normalized_bearings = K\[keypoint_sample; ones(1, 3)];
    for ii = 1:3
        normalized_bearings(:, ii) = normalized_bearings(:, ii) / ...
            norm(normalized_bearings(:, ii), 2);
    end
    poses = p3p(landmark_sample, normalized_bearings);
    R_C_W_guess = zeros(3, 3, 2);
    t_C_W_guess = zeros(3, 1, 2);
    for ii = 0:1
        R_W_C_ii = real(poses(:, (2+ii*4):(4+ii*4)));
        t_W_C_ii = real(poses(:, (1+ii*4)));
        R_C_W_guess(:,:,ii+1) = R_W_C_ii';
        t_C_W_guess(:,:,ii+1) = -R_W_C_ii'*t_W_C_ii;
    end
    
    % Count inliers:
    projected_points = projectPoints(...
        (R_C_W_guess(:,:,1) * p_W_landmarks) + ...
        repmat(t_C_W_guess(:,:,1), ...
        [1 size(p_W_landmarks, 2)]), K);
    difference = query_keypoints - projected_points;
    errors = sum(difference.^2, 1);
    is_inlier = errors < pixel_tolerance^2;
        
    projected_points = projectPoints(...
        (R_C_W_guess(:,:,2) * p_W_landmarks) + ...
        repmat(t_C_W_guess(:,:,2), ...
        [1 size(p_W_landmarks, 2)]), K);
    difference = query_keypoints - projected_points;
    errors = sum(difference.^2, 1);
    alternative_is_inlier = errors < pixel_tolerance^2;
    if nnz(alternative_is_inlier) > nnz(is_inlier)
        is_inlier = alternative_is_inlier;
        R_best = R_C_W_guess(:,:,2);
        t_best = t_C_W_guess(:,:,2);
    else
        R_best = R_C_W_guess(:,:,1);
        t_best = t_C_W_guess(:,:,1);
    end
        
    if nnz(is_inlier) > max_num_inliers && nnz(is_inlier) >= 6
        max_num_inliers = nnz(is_inlier);        
        inlier_mask = is_inlier;
        R_C_W = R_best;
        t_C_W = t_best;
    end
    
    max_num_inliers_history(i) = max_num_inliers;
end

% Perfoming smoothing...
if max_num_inliers == 0
    R_C_W = [];
    t_C_W = [];
else
    % Take only the inliers
    p_W_landmarks = p_W_landmarks(:,inlier_mask>0);
    query_keypoints = query_keypoints(:,inlier_mask>0);
    % Perform another iteration of RANSAC process
    [landmark_sample, idx] = datasample(...
        p_W_landmarks, s, 2, 'Replace', false);
    keypoint_sample = query_keypoints(:, idx);
    
    normalized_bearings = K\[keypoint_sample; ones(1, 3)];
    for ii = 1:3
        normalized_bearings(:, ii) = normalized_bearings(:, ii) / ...
            norm(normalized_bearings(:, ii), 2);
    end
    poses = p3p(landmark_sample, normalized_bearings);
    R_C_W_guess = zeros(3, 3, 2);
    t_C_W_guess = zeros(3, 1, 2);
    for ii = 0:1
        R_W_C_ii = real(poses(:, (2+ii*4):(4+ii*4)));
        t_W_C_ii = real(poses(:, (1+ii*4)));
        R_C_W_guess(:,:,ii+1) = R_W_C_ii';
        t_C_W_guess(:,:,ii+1) = -R_W_C_ii'*t_W_C_ii;
    end
    
    % Count inliers:
    projected_points = projectPoints(...
        (R_C_W_guess(:,:,1) * p_W_landmarks) + ...
        repmat(t_C_W_guess(:,:,1), ...
        [1 size(p_W_landmarks, 2)]), K);
    difference = query_keypoints - projected_points;
    errors = sum(difference.^2, 1);
    is_inlier = errors < pixel_tolerance^2;
        
    projected_points = projectPoints(...
        (R_C_W_guess(:,:,2) * p_W_landmarks) + ...
        repmat(t_C_W_guess(:,:,2), ...
        [1 size(p_W_landmarks, 2)]), K);
    difference = query_keypoints - projected_points;
    errors = sum(difference.^2, 1);
    alternative_is_inlier = errors < pixel_tolerance^2;
    if nnz(alternative_is_inlier) > nnz(is_inlier)
        is_inlier = alternative_is_inlier;
        R_best = R_C_W_guess(:,:,2);
        t_best = t_C_W_guess(:,:,2);
    else
        R_best = R_C_W_guess(:,:,1);
        t_best = t_C_W_guess(:,:,1);
    end
        
    if nnz(is_inlier) > max_num_inliers && nnz(is_inlier) >= 6
        max_num_inliers = nnz(is_inlier);        
        inlier_mask = is_inlier;
        R_C_W = R_best;
        t_C_W = t_best;
    end
    
end


end
