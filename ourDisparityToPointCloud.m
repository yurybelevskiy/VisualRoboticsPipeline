function [points, output_keypoints, intensities] = ourDisparityToPointCloud(...
    disp_img, K, baseline, left_img, input_keypoints)
% points should be 3xN and intensities 1xN, where N is the amount of pixels
% which have a valid disparity. I.e., only return points and intensities
% for pixels of left_img which have a valid disparity estimate! The i-th
% intensity should correspond to the i-th point.

[X,Y] = meshgrid(1:size(disp_img,2),1:size(disp_img,1));

px_left = [Y(:) X(:) ones(numel(disp_img), 1)]';

px_right = px_left;
px_right(2, :) = px_right(2, :) - disp_img(:)';

num_keypoints = size(input_keypoints,2);
index = zeros(1,num_keypoints);
for k = 1:num_keypoints
    row = ceil(input_keypoints(2,k));
    column = ceil(input_keypoints(1,k));
    value = row+(column-1)*size(disp_img,1);
    % Check if the value is already stored
    if isempty(find(value==index(1:k), 1))
        index(k) = value;
    end
end
% Take zeros out
index = index(index~=0);

positions = zeros(1,size(disp_img(:),1));
index = index(disp_img(index)>0);
positions(index) = 1; 
positions = logical(positions);
px_left = px_left(:, positions>0);
px_right = px_right(:, positions>0);

% Switch from (row, col, 1) to (u, v, 1)
px_left(1:2, :) = flipud(px_left(1:2,:)); 
px_right(1:2, :) = flipud(px_right(1:2, :));

bv_left = K^-1 * px_left; 
bv_right = K^-1 * px_right; 

temp_points = zeros(size(px_left));

b = [baseline; 0; 0];
for i = 1:size(px_left, 2)
    A = [bv_left(:, i) -bv_right(:, i)];
    x = (A' * A) \ (A' * b);
    temp_points(:, i) = bv_left(:, i) * x(1);
end

% Recreate the order of the points
points = zeros(3,size(index,2));
output_keypoints = zeros(2,size(index,2));
sorted_index = sort(index);
for k = 1:size(index,2)
    points(:,k) = temp_points(:,sorted_index == index(k));
    output_keypoints(:,k) = input_keypoints(:,sorted_index == index(k));
end

intensities = left_img(disp_img(:)' > 0);

end

