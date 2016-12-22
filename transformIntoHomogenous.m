function p_homogenous = transformIntoHomogenous(p)
% Function that gives back the homogenous coordinates of the points  

[rows, npts] = size(p);
p_homogenous = ones(rows+1, npts);
p_homogenous(1:rows,:) = p;

end

