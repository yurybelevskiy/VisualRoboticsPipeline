function [  ] = plotMatched( img, p1, p2, ransac )

figure;
imshow(img); hold on;
plot(p1(2,:),p1(1,:),'rx','linewidth',1.5); 
plot(p2(2,:),p2(1,:),'go','linewidth',1.5);

x_from = p2(1,:);
x_to = p1(1,:);
y_from = p2(2,:);
y_to = p1(2,:);

plot([y_from;y_to],[x_from;x_to],'y-','linewidth',1.25);

if ransac
    title('With ransac');
else
    title('Without ransac');
end

hold off;


end

