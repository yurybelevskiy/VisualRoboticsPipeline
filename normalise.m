function norm_p = normalise(p)
    
    rows = size(p,1);
    norm_p = p;

    % Find the indices of the points that are not at infinity
    index = find(abs(p(rows,:)) > eps);

    % Normalise points
    for r = 1:rows-1
        norm_p(r,index) = p(r,index)./p(rows,index);
    end
    norm_p(rows,index) = 1;
end