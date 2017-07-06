% Deep GP-based non-linear IRL algorithm.

function irl_result = deepgpirlrun(algorithm_params,mdp_data,mdp_model,...
    feature_data,example_samples,~,verbosity)

% algorithm_params - parameters of the DGP-IRL algorithm.
% mdp_data - definition of the MDP to be solved.
% example_samples - cell array containing examples.

% Fill in default parameters.
model = deepgpirldefaultparams(algorithm_params);
model.mdp_data = mdp_data;
% Set random seed.
rand('seed',model.seed);
randn('seed',model.seed);

% Get state-action counts and initial state distributions.
[mu_sa,init_s] = gpirlgetstatistics(example_samples,mdp_data);
% Constants.
model.init_s = init_s;
states = size(mu_sa,1);
mu_s = sum(mu_sa,2);
model.mu_sa = mu_sa;

%% Construct X, W
if model.all_features,
    X = feature_data.splittable;
elseif model.true_features,
    X = true_features;
else
    X = eye(states);
end;

ind_sel = [];
for i = 1:size(X,2)
    if var(X(:,i)) > 0
        ind_sel = [ind_sel,i];
    end
end
X = X(:,ind_sel);
model.X = X;

if strcmp(model.inducing_pts,'examples'),
    % Select just the example states.
    s_u = find(mu_s)';
elseif strcmp(model.inducing_pts,'examplesplus'),
    % Select example states, plus random states to bring up to desired total.
    s_u = find(mu_s)';
    if length(s_u) < model.inducing_pts_count,
        other_states = find(~mu_s)';
        other_states = other_states(randperm(length(other_states)));
        s_u = [s_u other_states(...
            1:(model.inducing_pts_count-length(s_u)))];
        s_u = sort(s_u);
    end;
elseif strcmp(model.inducing_pts,'random'),
    % Select random states.
    s_u = randperm(states);
    s_u = sort(s_u(1:model.inducing_pts_count));
else
    % Select all states.
    s_u = 1:states;
end

model.W = model.X(s_u,:);
model.K = length(s_u);
model.N = states;
model.inpR = size(X,2);

%% Initialize model parameters: initial_r, u,E

if ~isempty(model.initial_r),
    % Use provided initialization.
    if ~iscell(model.initial_r),
        initial_r = mean(model.initial_r,2);
    else
        initial_r = model.initial_r;
        rmat = [];
        for i=1:length(r),
            rmat = [rmat mean(r{i},2)];
        end;
        initial_r = mean(rmat,2);
    end
else
    initial_r = rand(states,model.num_init_r);
end

%% Initialization

nnvec = zeros(model.num_init_r);
modelcell = cell(model.num_init_r);
for i = 1:model.num_init_r
    model.u = initial_r(s_u,i);
    model.initial_r = initial_r(:,i);
    model = dgpInitModel(model);
    params = dgpExtractParams(model);
    model = dgpUpdateStats(model,params);
    
    % Create anonymous function.
    fun = @(x)dgpObjGradient(x,model);
    options.Method = 'pcg';
    options.TolX = 1e-5;
    options.TolFun = 1e-5;
    options.MaxIter = 50;
    options.MaxFunEvals = 50;
    tic;
    [best_x,best_nll] = minFunc(fun,params,options);
    time = toc;
    if verbosity ~= 0,
        fprintf(1,'Completed initialization run in %f s, LL = %f\n',time,-best_nll);
    end;
    model = dgpUpdateStats(model,best_x);
    nnvec(i) = best_nll;
    modelcell{i} = model;
end

[~,bind] = min(nnvec);
model = modelcell{bind};
params = dgpExtractParams(model);
model = dgpUpdateStats(model,params);

% Create anonymous function.
fun = @(x)dgpObjGradient(x,model);

%% Optimization
options.TolX = 1e-9;
options.TolFun = 1e-9;
for restart_ii = 1:model.restart_num
    options.MaxIter = model.restart_iter;
    options.MaxFunEvals = model.restart_iter;
    tic;
    [best_x,best_nll] = minFunc(fun,params,options);
    time = toc;
    if verbosity ~= 0,
        fprintf(1,'Completed run of %d in %f s, LL = %f\n',restart_ii,time,-best_nll);
    end;
    
    model = dgpUpdateStats(model,best_x);
    fun = @(x)dgpObjGradient(x,model);
    params = dgpExtractParams(model);
end

r = model.KDZ_invKZZ*model.u;
r = repmat(r,1,mdp_data.actions);
soln = feval([mdp_model 'solve'],mdp_data,r);
v = soln.v;
q = soln.q;
p = soln.p;

% Construct returned structure.
irl_result = struct('r',r,'v',v,'p',p,'q',q,'model_itr',{{model}},...
    'r_itr',{{r}},'model_r_itr',{{r}},'p_itr',{{p}},'model_p_itr',{{p}});
