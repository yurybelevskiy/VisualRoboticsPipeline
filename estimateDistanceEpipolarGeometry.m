function [ distance ] = estimateDistanceEpipolarGeometry (F,p1,p2)
% l (3xS) : contains [a,b,c] of the epipolar line (ay+bx+c=0);
% p (3xN) : point whose distance from l is computed (homogeneous coord).

% Epipolar lines
l1 = F'*p2;
l2 = F*p1;

N = size(p1,2);
% Vector of distances
d1 = zeros(N,1);
d2 = zeros(N,1);

for k = 1:N
    d1(k) = calculateDistancePointLine(l1(:,k),p1(:,k));
    d2(k) = calculateDistancePointLine(l2(:,k),p2(:,k));
end
distance = d1.^2+d2.^2;
end

function distance = calculateDistancePointLine(l,p)
    % Angular coefficient and offset of l
    m = -l(2)./l(1);
    q = -l(3)./l(1);
    % Angular coefficient and offset of the perpendicular line to l
    m_p = -1./m;
    q_p = p(1)-m_p.*p(2);
    % Intercepting point of the two lines
    x_intercept = (q-q_p)./(m_p-m);
    y_intercept = m.*x_intercept+q;
    pixel = [x_intercept;y_intercept;1]; %normc([x_intercept;y_intercept;1]);
    % Distance between the target point and the interception point
    distance = sqrt((p(2)-pixel(1)).^2 + (p(1)-pixel(2)).^2);
end