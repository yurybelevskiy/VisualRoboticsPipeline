function [ inlier_mask, best_F ] = RANSAC_8pointAlg( p1, p2 )
% This function applies the RANSAC to the 8 point algorithm in order to
% elimate outliers form the estimation of the camera pose. The working
% principle is based on the distance of the point from the epipolar line.
% Here, the estimated fundamental matrix F and the tolerance of 1 pixel are
% used.

% Input:  - p1 (3xN): homogeneous coordinates of points on frame n. 1;
%         - P2 (3XN): homogeneous coordinates of points on frame n. 2;
%
% Output: - inlier_mask : logical mask to detect inliers (1:inlier,
%                         0:outlier);
%          - best_F (3x3): best fundamental matrix found.
%
% NOTE: The input points have already been matched.

% Start of the RANSAC algorithm
% Initialization: number of iteration
p = 0.99;                   % Probability that a sample is free from outliers
e = 0.55;                   % Probability that a point is an outlier
s = 8;                      % Number of point of correspondence
num_iterations = log(1-p)/log(1-(1-e).^s);
pixel_tolerance = 4;        % Higher tolerance for initialization

% Initilization and pre-allocation of the varibles
inlier_mask = zeros(1, size(p1, 2));
max_num_inliers_history = zeros(1, ceil(num_iterations));
max_num_inliers = 0;

% Normalization of points is done in the function
% "fundamentalEightPoint_normalized"
for i = 1:num_iterations
    % Sample the points in the first frame and then find the correspondent
    % ones in the second frame.
    [p1_sample, idx] = datasample(p1, s, 2, 'Replace', false);
    p2_sample = p2(:,idx);
    
    F_est = fundamentalEightPoint_normalized(p1_sample,p2_sample);
    distance = estimateDistanceEpipolarGeometry(F_est,p1,p2);
    is_inlier = distance < pixel_tolerance^2;
    
    if nnz(is_inlier) > max_num_inliers && nnz(is_inlier) >= 6
        max_num_inliers = nnz(is_inlier);
        if max_num_inliers > max(max_num_inliers_history(:))
            best_F = F_est;
        end
        inlier_mask = is_inlier;
    end
    
    max_num_inliers_history(i) = max_num_inliers;
end

if max_num_inliers == 0
    inlier_mask = [];
    best_F = [];
end

end



