% Draw single objectworld with specified reward function.
function binaryworlddraw(r,p,g,mdp_params,mdp_data,feature_data,model)

% Use gridworld drawing function to draw paths and reward function.
if nargin == 5,
    gridworlddraw(r,p,g,mdp_params,mdp_data);
elseif nargin == 6,
    gridworlddraw(r,p,g,mdp_params,mdp_data,feature_data);
elseif nargin == 7,
    gridworlddraw(r,p,g,mdp_params,mdp_data,feature_data,model);
end;

% Initialize colors.
shapeColors = [242,60,30;30,110,242]/255;
edgeColors = [242,60,30;30,110,242]/255;

if iscell(g),
    % This means p is crop.
    crop = p;
else
    crop = [1 mdp_params.n; 1 mdp_params.n];
end;

% Draw objects.
for x = 1:mdp_params.n
    for y = 1:mdp_params.n
        xd = x-crop(1,1)+1;
        yd = y-crop(2,1)+1;
        % Draw the object.
        rectangle('Position',[xd-0.65,yd-0.65,0.3,0.3],'Curvature',[1,1],...
            'FaceColor',shapeColors(mdp_data.map(x,y)+1,:),...
            'EdgeColor',edgeColors(mdp_data.map(x,y)+1,:),'LineWidth',2);
    end
end
