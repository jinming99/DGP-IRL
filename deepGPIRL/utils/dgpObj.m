function ll = dgpObj(params,model)
%% NOTE: THIS IS NEGATIVE LOGLIKELIHOOD!
model = dgpUpdateStats(model,params);
[~,actions] = size(model.mu_sa);
yrwd = model.KDZ_invKZZ*model.u;
% Run value iteration to get policy.
[~,~,policy,logpolicy] = linearvalueiteration(model.mdp_data,repmat(yrwd,1,actions));

K = model.K; R = model.R; N = model.N; inpR = model.inpR;
% Compute value by adding up log example probabilities.
L_M = sum(sum(logpolicy.*model.mu_sa));
L_G = -0.5*model.u'*model.invKZZ_u-0.5*K*log(2*pi)-model.halfLogDetKZZ;
L_DBV = -0.5*N*R*log(2*pi/model.lambda)-0.5*model.lambda*(sum(sum(model.E2sum'.*model.KKKK_WX))...
    +model.R*trace(model.SigmaB)+sum(sum(model.KKKK_WX'.*(model.R*model.Sigmav+model.VVTsum)))...
    -2*sum(sum(model.KXW_invKWW'.*model.VDsum)))-model.R*model.halfLogDetKWW-...
    0.5*sum(sum(model.invKWW'.*(model.R*model.Sigmav+model.VVTsum)))+model.R*model.halfLogDetSigmav;
ll = L_M+L_G+L_DBV+...
        dgpirlhpprior(gpirlhpxform(model.Lambda1,[],model.ard_xform,3),model.ard_prior,model.ard_prior_wt,model.ard_xform,model)+...
        dgpirlhpprior(gpirlhpxform(model.Lambda2,[],model.ard_xform,3),model.ard_prior,model.ard_prior_wt,model.ard_xform,model);
ll = -ll;
