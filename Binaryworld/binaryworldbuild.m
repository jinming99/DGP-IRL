% Construct the Objectworld MDP structures.
function [mdp_data,r,feature_data,true_feature_map] = binaryworldbuild(mdp_params)

% mdp_params - parameters of the objectworld:
%       seed (0) - initialization for random seed
%       n (32) - number of cells along each axis
%       placement_prob (0.05) - probability of placing object in each cell
%       c1 (2) - number of primary "colors"
%       c2 (2) - number of secondary "colors"
%       determinism (1.0) - probability of correct transition
%       discount (0.9) - temporal discount factor to use
% mdp_data - standard MDP definition structure with object-world details:
%       states - total number of states in the MDP
%       actions - total number of actions in the MDP
%       discount - temporal discount factor to use
%       sa_s - mapping from state-action pairs to states
%       sa_p - mapping from state-action pairs to transition probabilities
%       map1 - mapping from states to c1 colors
%       map2 - mapping from states to c2 colors
%       c1array - array of locations by c1 colors
%       c2array - array of locations by c2 colors
% r - mapping from state-action pairs to rewards

% Fill in default parameters.
mdp_params = binaryworlddefaultparams(mdp_params);

% Set random seed.
rand('seed',mdp_params.seed);

% Build action mapping.
sa_s = zeros(mdp_params.n^2,5,5);
sa_p = zeros(mdp_params.n^2,5,5);
for y=1:mdp_params.n,
    for x=1:mdp_params.n,
        s = (y-1)*mdp_params.n+x;
        successors = zeros(1,1,5);
        successors(1,1,1) = s;
        successors(1,1,2) = (min(mdp_params.n,y+1)-1)*mdp_params.n+x;
        successors(1,1,3) = (y-1)*mdp_params.n+min(mdp_params.n,x+1);
        successors(1,1,4) = (max(1,y-1)-1)*mdp_params.n+x;
        successors(1,1,5) = (y-1)*mdp_params.n+max(1,x-1);
        sa_s(s,:,:) = repmat(successors,[1,5,1]);
        sa_p(s,:,:) = reshape(eye(5,5)*mdp_params.determinism + ...
            (ones(5,5)-eye(5,5))*((1.0-mdp_params.determinism)/4.0),...
            1,5,5);
    end;
end;

% Construct map.
map = rand(mdp_params.n,mdp_params.n)<mdp_params.placeblue_prob;%zeros(mdp_params.n^2,1);

% Create MDP data structure.
mdp_data = struct(...
    'states',mdp_params.n^2,...
    'actions',5,...
    'discount',mdp_params.discount,...
    'determinism',mdp_params.determinism,...
    'sa_s',sa_s,...
    'sa_p',sa_p,...
    'map',map);

% Construct feature map and rewards
R_SCALE = 5;
splittable = zeros(mdp_params.n^2,9);
r = zeros(mdp_params.n^2,1);
map_vec = map(:)';
indall = 1:9;
for y=1:mdp_params.n,
    for x=1:mdp_params.n,
        s = (y-1)*mdp_params.n+x;
        indx(1) = x-1; indy(1) = y-1;
        indx(2) = x-1; indy(2) = y;
        indx(3) = x-1; indy(3) = y+1;
        indx(4) = x; indy(4) = y-1;
        indx(5) = x; indy(5) = y;
        indx(6) = x; indy(6) = y+1;
        indx(7) = x+1; indy(7) = y-1;
        indx(8) = x+1; indy(8) = y;
        indx(9) = x+1; indy(9) = y+1;
        
        indsel = (indy-1)*mdp_params.n+indx;
        indselout = indall(indx==0|indx==(mdp_params.n+1)|indy==0|indy==(mdp_params.n+1));
        indselin = setdiff(indall,indselout);
        splittable(s,indselin) = map_vec(indsel(indselin));
        splittable(s,indselout) = 0;
        if sum(splittable(s,:)) == 4
            r(s) = 1;
        elseif sum(splittable(s,:)) == 5
            r(s) = -1;
        else
            r(s) = 0;
        end
    end;
end;

% Construct adjacency table.
stateadjacency = sparse([],[],[],mdp_data.states,mdp_data.states,...
    mdp_data.states*mdp_data.actions);
for s=1:mdp_data.states,
    for a=1:mdp_data.actions,
        stateadjacency(s,mdp_data.sa_s(s,a,1)) = 1;
    end;
end;
% Return feature data structure.
if mdp_params.continuous
    feature_data = struct('stateadjacency',stateadjacency,'splittable',splittable);
    % Construct true feature map.
    true_feature_map = splittable;
else
    splittable_cont = splittable;
    splittable = [];
    colorvec = [1,0;0,1];
    for i = size(splittable_cont,1)
        splitvec = [];
        for j = size(splittable_cont,2)
            splitvec = [splitvec,colorvec(splittable_cont(i,j)+1,:)];
        end
        splittable = [splittable;splitvec];
    end
    feature_data = struct('stateadjacency',stateadjacency,'splittable',splittable);
    % Construct true feature map.
    true_feature_map = splittable;
end



r = repmat(r*R_SCALE,1,5);