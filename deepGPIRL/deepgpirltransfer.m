% Transfer learned reward function to a new state space.
function irl_result = deepgpirltransfer(prev_result,mdp_data,mdp_model,...
                                    feature_data,~,verbosity)

model = prev_result.model_itr{end};

% To transfer the result to the new state space, we must first compute
% the latent space D for input X
Xnew = feature_data.splittable;
if isfield(model,'ind_sel')
    Xnew = Xnew(:,model.ind_sel);
end
KDstarZ = dgpirlkernel(model,Xnew);

r = repmat(KDstarZ*model.invKZZ*model.u,1,mdp_data.actions);
% Solve MDP.
soln = feval([mdp_model 'solve'],mdp_data,r);
v = soln.v;
q = soln.q;
p = soln.p;

% Build IRL result.
irl_result = struct('r',r,'v',v,'p',p,'q',q,'model_itr',{{model}},...
                    'r_itr',{{r}},'model_r_itr',{{r}},'p_itr',{{p}},'model_p_itr',{{p}});

