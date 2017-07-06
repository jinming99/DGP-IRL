function [ll,grad] = dgpObjGradient(params,model)
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

%% calculate the gradients
if nargout >= 2,
    
    D = linearmdpfrequency(model.mdp_data,struct('p',policy),model.init_s);
    dLM_dyrwd = sum(model.mu_sa,2) - D; %dL_M/dyrwd
    dLM_du = model.KDZ_invKZZ'*dLM_dyrwd; %dL_M/du
    dLG_du = -model.invKZZ_u; %dL_G/du
    
    % create dyrwd_dKDZ: checked!
    idy = repmat(1:N,1,K);
    idx = kron((0:(K-1))*N,ones(1,N))+idy;
    vv = kron(model.invKZZ_u',ones(1,N));
    dyrwd_dKDZ = sparse(idx,idy,vv,K*N,N);
    
    % create dKDZ_dD: checked!
    dKDZ_dD = ardjitkernelGradXaug(model.D,model.Z,model.Lambda2,model.beta2,model.sigma2,1,R);
    
    % create dD_dE: checked!
    idx = kron((0:(R-1))*K,ones(1,K*N))+repmat(kron(1:K,ones(1,N)),1,R);
    idy = kron((0:(R-1))*N,ones(1,K*N))+repmat(1:N,1,K*R);
    vv = repmat(model.KXW_invKWW(:),1,R);
    dD_dE = sparse(idx,idy,vv,R*K,N*R);
    
    dLM_dE = dD_dE*dKDZ_dD*dyrwd_dKDZ*dLM_dyrwd; %dL_M/dE checked!
    
    dLDBV_dE = (2*model.lambda*(model.KWX_KXW_invKWW'*model.Gamma'*model.KWX*model.KXW_invKWW)...
        +(-0.5*model.lambda*(2*model.KKKK_WX+...
        2*(model.KWX_KXW_invKWW'*model.Gamma'*model.KWX*model.KWX'*model.Gamma*model.KWX_KXW_invKWW)))...
        -model.KWX_KXW_invKWW'*model.Gamma*model.KWW*model.Gamma*model.KWX_KXW_invKWW)*model.E;
    dLDBV_dE = dLDBV_dE(:);
    
    % For inducing inputs locations: Z checked!
    dKDZ_dZ = ardjitkernelGradXaug(model.D,model.Z,model.Lambda2,model.beta2,model.sigma2,2,R);
    dKZZ_dZ = ardjitkernelGradXaug(model.Z,[],model.Lambda2,model.beta2,model.sigma2,1,R);
    dyrwd_dZ = zeros(R*K,N);
    dLG_dZ = zeros(R*K,1);
    for i = 1:R
        for j = 1:K
            dKDZ_dZ_t = reshape(dKDZ_dZ(j+(i-1)*K,:),N,K);
            dKZZ_dZ_t = reshape(dKZZ_dZ(j+(i-1)*K,:),K,K);
            dyrwd_dZ(j+(i-1)*K,:) = (dKDZ_dZ_t*model.invKZZ_u-model.KDZ_invKZZ*dKZZ_dZ_t*model.invKZZ_u)';
            dLG_dZ(j+(i-1)*K,1) = 0.5*model.invKZZ_u'*dKZZ_dZ_t*model.invKZZ_u-0.5*sum(sum(model.invKZZ.*dKZZ_dZ_t));
        end
    end
    
    dLM_dZ = dyrwd_dZ*dLM_dyrwd;
    
    %% Hyperparameters:
    if ~model.use_num_diff
        % Layer 2: Lambda, sigma, beta
        dyrwd_dLambda2 = zeros(R,N); %TESTED
        dLG_dLambda2 = zeros(R,1); %TESTED
        dKDZ_dLambda2 = ardjitkernelGradLambda(model.D,model.Z,model.Lambda2,model.beta2,model.sigma2);
        dKZZ_dLambda2 = ardjitkernelGradLambda(model.Z,[],model.Lambda2,model.beta2,model.sigma2);
        for r = 1:R
            dyrwd_dLambda2(r,:) = (dKDZ_dLambda2(:,:,r)*model.invKZZ_u-model.KDZ_invKZZ*dKZZ_dLambda2(:,:,r)*model.invKZZ_u)';
            dLG_dLambda2(r) = 0.5*model.invKZZ_u'*dKZZ_dLambda2(:,:,r)*model.invKZZ_u-0.5*sum(sum(model.invKZZ.*dKZZ_dLambda2(:,:,r)));
        end
        dLM_dLambda2 = dyrwd_dLambda2*dLM_dyrwd;
        
        dKDZ_dsigma2 = ardjitkernelGrad_sigma(model.D,model.Z,model.Lambda2,model.beta2,model.sigma2);
        dKZZ_dsigma2 = ardjitkernelGrad_sigma(model.Z,[],model.Lambda2,model.beta2,model.sigma2);
        dyrwd_dsigma2 = (dKDZ_dsigma2*model.invKZZ_u-model.KDZ_invKZZ*dKZZ_dsigma2*model.invKZZ_u)'; %TESTED
        dLM_dsigma2 = dyrwd_dsigma2*dLM_dyrwd;
        dLG_dsigma2 = 0.5*model.invKZZ_u'*dKZZ_dsigma2*model.invKZZ_u-0.5*sum(sum(model.invKZZ.*dKZZ_dsigma2)); %TESTED
        
        dKDZ_dbeta2 = ardjitkernelGradBeta(model.D,model.Z,model.Lambda2,model.beta2,model.sigma2);
        dKZZ_dbeta2 = ardjitkernelGradBeta(model.Z,[],model.Lambda2,model.beta2,model.sigma2);
        dyrwd_dbeta2 = (dKDZ_dbeta2*model.invKZZ_u-model.KDZ_invKZZ*dKZZ_dbeta2*model.invKZZ_u)'; %TESTED
        dLM_dbeta2 = dyrwd_dbeta2*dLM_dyrwd;
        dLG_dbeta2 = 0.5*model.invKZZ_u'*dKZZ_dbeta2*model.invKZZ_u-0.5*sum(sum(model.invKZZ.*dKZZ_dbeta2));%TESTED
        
        % Layer 1: Lambda1 (CHECKED), sigma1(CHECKED), beta1
        dKWW_dLambda1 = ardjitkernelGradLambda(model.W,[],model.Lambda1,model.beta1,model.sigma1);
        dKXW_dLambda1 = ardjitkernelGradLambda(model.X,model.W,model.Lambda1,model.beta1,model.sigma1);
        dKXX_dLambda1 = ardjitkernelGradLambda(model.X,[],model.Lambda1,model.beta1,model.sigma1);
        dKWWWW_dLambda1 = zeros(K,K,inpR);
        dSigmaB_dLambda1 = zeros(N,N,inpR);
        dSigmav_dLambda1 = zeros(K,K,inpR);
        dKSv_dLambda1 = zeros(K,K,inpR);
        dKSvK_dLambda1 = zeros(K,K,inpR);
        dKVV_dLambda1 = zeros(inpR,1);
        dinvKVV_dLambda1 = zeros(inpR,1);
        dVKD_dLambda1 = zeros(inpR,1);
        dLDBV_dLambda1 = zeros(inpR,1);
        
        dD_dLambda1 = zeros(inpR,N*R);
        KSv = model.KKKK_WX*model.Sigmav;
        KSvK = model.KKKK_WX*model.Sigmav*model.KKKK_WX;
        for r = 1:inpR
            dKWWWW_dLambda1(:,:,r) = -model.invKWW*dKWW_dLambda1(:,:,r)*model.KKKK_WX+model.invKWW*dKXW_dLambda1(:,:,r)'*model.KXW_invKWW+...
                +model.KXW_invKWW'*dKXW_dLambda1(:,:,r)*model.invKWW-model.KKKK_WX*dKWW_dLambda1(:,:,r)*model.invKWW;
            dSigmaB_dLambda1(:,:,r) = (dKXX_dLambda1(:,:,r)-dKXW_dLambda1(:,:,r)*model.KXW_invKWW'...
                +model.KXW_invKWW*dKWW_dLambda1(:,:,r)*model.KXW_invKWW'-model.KXW_invKWW*dKXW_dLambda1(:,:,r)').*eye(N);
            dSigmav_dLambda1(:,:,r) = -model.Sigmav*(-model.invKWW*dKWW_dLambda1(:,:,r)*model.invKWW+model.lambda*dKWWWW_dLambda1(:,:,r))*model.Sigmav;
            dKSv_dLambda1(:,:,r) = dKWWWW_dLambda1(:,:,r)*model.Sigmav+model.KKKK_WX*dSigmav_dLambda1(:,:,r);
            dKSvK_dLambda1(:,:,r) = dKSv_dLambda1(:,:,r)*model.KKKK_WX+KSv*dKWWWW_dLambda1(:,:,r);
            dKVV_dLambda1(r) = model.lambda^2*trace((dKSvK_dLambda1(:,:,r)*KSv'+KSvK*dKSv_dLambda1(:,:,r)')*model.E2sum);
            dVKD_dLambda1(r) = model.lambda^2*trace(dKSvK_dLambda1(:,:,r)*model.E2sum);
            dinvKVV_dLambda1(r) = model.lambda^2*trace((dKSv_dLambda1(:,:,r)*model.invKWW*KSv'+...should be -
                KSv*model.invKWW*dKWW_dLambda1(:,:,r)*model.invKWW*KSv+KSv*model.invKWW*dKSv_dLambda1(:,:,r)')*model.E2sum);
            dLDBV_dLambda1(r) = -0.5*model.lambda*((trace(dKWWWW_dLambda1(:,:,r)*model.E2sum)+trace(R*dSigmaB_dLambda1(:,:,r))+...
                trace(R*dKSv_dLambda1(:,:,r)))+dKVV_dLambda1(r))+dVKD_dLambda1(r)-0.5*R*sum(sum(model.invKWW'.*dKWW_dLambda1(:,:,r)))...
                -0.5*R*(trace(-model.invKWW*dKWW_dLambda1(:,:,r)*model.invKWW*model.Sigmav+model.invKWW*dSigmav_dLambda1(:,:,r)))...
                -0.5*model.lambda^2*trace((dKSv_dLambda1(:,:,r)*model.invKWW*KSv'-KSv*model.invKWW*dKWW_dLambda1(:,:,r)*model.invKWW*KSv'+...
                KSv*model.invKWW*dKSv_dLambda1(:,:,r)')*model.E2sum)+...
                0.5*R*trace((model.invKWW+model.lambda*model.KKKK_WX)*dSigmav_dLambda1(:,:,r));
            dD_dLambda1r = dKXW_dLambda1(:,:,r)*model.invKWW*model.E - model.KXW_invKWW*dKWW_dLambda1(:,:,r)*model.invKWW*model.E;
            dD_dLambda1(r,:) = dD_dLambda1r(:);
        end
        dLM_dLambda1 = dD_dLambda1*dKDZ_dD*dyrwd_dKDZ*dLM_dyrwd;
        
        
        dKWW_dsigma1 = ardjitkernelGrad_sigma(model.W,[],model.Lambda1,model.beta1,model.sigma1);
        dKXW_dsigma1 = ardjitkernelGrad_sigma(model.X,model.W,model.Lambda1,model.beta1,model.sigma1);
        dKXX_dsigma1 = ardjitkernelGrad_sigma(model.X,[],model.Lambda1,model.beta1,model.sigma1);
        KSv = model.KKKK_WX*model.Sigmav;
        KSvK = model.KKKK_WX*model.Sigmav*model.KKKK_WX;
        dKWWWW_dsigma1 = -model.invKWW*dKWW_dsigma1*model.KKKK_WX+model.invKWW*dKXW_dsigma1'*model.KXW_invKWW+...
            +model.KXW_invKWW'*dKXW_dsigma1*model.invKWW-model.KKKK_WX*dKWW_dsigma1*model.invKWW;
        dSigmaB_dsigma1 = (dKXX_dsigma1-dKXW_dsigma1*model.KXW_invKWW'...
            +model.KXW_invKWW*dKWW_dsigma1*model.KXW_invKWW'-model.KXW_invKWW*dKXW_dsigma1').*eye(N);
        dSigmav_dsigma1 = -model.Sigmav*(-model.invKWW*dKWW_dsigma1*model.invKWW+model.lambda*dKWWWW_dsigma1)*model.Sigmav;
        dKSv_dsigma1 = dKWWWW_dsigma1*model.Sigmav+model.KKKK_WX*dSigmav_dsigma1;
        dKSvK_dsigma1 = dKSv_dsigma1*model.KKKK_WX+KSv*dKWWWW_dsigma1;
        dKVV_dsigma1 = model.lambda^2*trace((dKSvK_dsigma1*KSv'+KSvK*dKSv_dsigma1')*model.E2sum);
        dVKD_dsigma1 = model.lambda^2*trace(dKSvK_dsigma1*model.E2sum);
        dinvKVV_dsigma1 = model.lambda^2*trace((dKSv_dsigma1*model.invKWW*KSv'+...
            KSv*model.invKWW*dKWW_dsigma1*model.invKWW*KSv+KSv*model.invKWW*dKSv_dsigma1')*model.E2sum);
        dLDBV_dsigma1 = -0.5*model.lambda*((trace(dKWWWW_dsigma1*model.E2sum)+trace(R*dSigmaB_dsigma1)+...
            trace(R*dKSv_dsigma1))+dKVV_dsigma1)+dVKD_dsigma1-0.5*R*sum(sum(model.invKWW'.*dKWW_dsigma1))...
            -0.5*R*(trace(-model.invKWW*dKWW_dsigma1*model.invKWW*model.Sigmav+model.invKWW*dSigmav_dsigma1))...
            -0.5*model.lambda^2*trace((dKSv_dsigma1*model.invKWW*KSv'-KSv*model.invKWW*dKWW_dsigma1*model.invKWW*KSv'+...
            KSv*model.invKWW*dKSv_dsigma1')*model.E2sum)+...
            0.5*R*trace((model.invKWW+model.lambda*model.KKKK_WX)*dSigmav_dsigma1);
        dD_dsigma1r = dKXW_dsigma1*model.invKWW*model.E - model.KXW_invKWW*dKWW_dsigma1*model.invKWW*model.E;
        dD_dsigma1 = reshape(dD_dsigma1r(:),1,length(dD_dsigma1r(:)));
        dLM_dsigma1 = dD_dsigma1*dKDZ_dD*dyrwd_dKDZ*dLM_dyrwd;%TESTED
        
        
        dKWW_dbeta1 = ardjitkernelGradBeta(model.W,[],model.Lambda1,model.beta1,model.sigma1);
        dKXW_dbeta1 = ardjitkernelGradBeta(model.X,model.W,model.Lambda1,model.beta1,model.sigma1);
        dKXX_dbeta1 = ardjitkernelGradBeta(model.X,[],model.Lambda1,model.beta1,model.sigma1);
        KSv = model.KKKK_WX*model.Sigmav;
        KSvK = model.KKKK_WX*model.Sigmav*model.KKKK_WX;
        dKWWWW_dbeta1 = -model.invKWW*dKWW_dbeta1*model.KKKK_WX+model.invKWW*dKXW_dbeta1'*model.KXW_invKWW+...
            +model.KXW_invKWW'*dKXW_dbeta1*model.invKWW-model.KKKK_WX*dKWW_dbeta1*model.invKWW;
        dSigmaB_dbeta1 = (dKXX_dbeta1-dKXW_dbeta1*model.KXW_invKWW'...
            +model.KXW_invKWW*dKWW_dbeta1*model.KXW_invKWW'-model.KXW_invKWW*dKXW_dbeta1').*eye(N);
        dSigmav_dbeta1 = -model.Sigmav*(-model.invKWW*dKWW_dbeta1*model.invKWW+model.lambda*dKWWWW_dbeta1)*model.Sigmav;
        dKSv_dbeta1 = dKWWWW_dbeta1*model.Sigmav+model.KKKK_WX*dSigmav_dbeta1;
        dKSvK_dbeta1 = dKSv_dbeta1*model.KKKK_WX+KSv*dKWWWW_dbeta1;
        dKVV_dbeta1 = model.lambda^2*trace((dKSvK_dbeta1*KSv'+KSvK*dKSv_dbeta1')*model.E2sum);
        dVKD_dbeta1 = model.lambda^2*trace(dKSvK_dbeta1*model.E2sum);
        dinvKVV_dbeta1 = model.lambda^2*trace((dKSv_dbeta1*model.invKWW*KSv'+...
            KSv*model.invKWW*dKWW_dbeta1*model.invKWW*KSv+KSv*model.invKWW*dKSv_dbeta1')*model.E2sum);
        dLDBV_dbeta1 = -0.5*model.lambda*((trace(dKWWWW_dbeta1*model.E2sum)+trace(R*dSigmaB_dbeta1)+...
            trace(R*dKSv_dbeta1))+dKVV_dbeta1)+dVKD_dbeta1-0.5*R*sum(sum(model.invKWW'.*dKWW_dbeta1))...
            -0.5*R*(trace(-model.invKWW*dKWW_dbeta1*model.invKWW*model.Sigmav+model.invKWW*dSigmav_dbeta1))...
            -0.5*model.lambda^2*trace((dKSv_dbeta1*model.invKWW*KSv'-KSv*model.invKWW*dKWW_dbeta1*model.invKWW*KSv'+...
            KSv*model.invKWW*dKSv_dbeta1')*model.E2sum)+...
            0.5*R*trace((model.invKWW+model.lambda*model.KKKK_WX)*dSigmav_dbeta1);
        dD_dbeta1r = dKXW_dbeta1*model.invKWW*model.E - model.KXW_invKWW*dKWW_dbeta1*model.invKWW*model.E;
        dD_dbeta1 = reshape(dD_dbeta1r(:),1,length(dD_dbeta1r(:)));
        dLM_dbeta1 = dD_dbeta1*dKDZ_dD*dyrwd_dKDZ*dLM_dyrwd;%TESTED
        
        % lambda: CHECKED
        %dL_dlambda = 0.5*N*R/model.lambda-0.5*(R*trace(model.SigmaB)+sum(sum(model.Gsum.*model.KKKK_WX)));
        
        %% Transform hyperparameter gradients
        dL_dLambda1 = gpirlhpxform(gpirlhpxform(model.Lambda1,[],model.ard_xform,3),dLM_dLambda1+dLDBV_dLambda1,model.ard_xform,2)+dgpirlhppriorgrad(gpirlhpxform(model.Lambda1,[],model.ard_xform,3),model.ard_prior,model.ard_prior_wt,model.ard_xform,model);
        dL_dsigma1 = gpirlhpxform(gpirlhpxform(model.sigma1,[],model.ard_xform,3),dLM_dsigma1+dLDBV_dsigma1,model.ard_xform,2);
        dL_dbeta1 = gpirlhpxform(gpirlhpxform(model.beta1,[],model.ard_xform,3),dLM_dbeta1+dLDBV_dbeta1,model.ard_xform,2);
        dL_dLambda2 = gpirlhpxform(gpirlhpxform(model.Lambda2,[],model.ard_xform,3),dLM_dLambda2+dLG_dLambda2,model.ard_xform,2)+dgpirlhppriorgrad(gpirlhpxform(model.Lambda2,[],model.ard_xform,3),model.ard_prior,model.ard_prior_wt,model.ard_xform,model);
        dL_dsigma2 = gpirlhpxform(gpirlhpxform(model.sigma2,[],model.ard_xform,3),dLM_dsigma2+dLG_dsigma2,model.ard_xform,2);
        dL_dbeta2 = gpirlhpxform(gpirlhpxform(model.beta2,[],model.ard_xform,3),dLM_dbeta2+dLG_dbeta2,model.ard_xform,2);
        %dL_dlambda = gpirlhpxform(model.lambda,dL_dlambda,model.ard_xform,2);
        eps = 1e-6;
        if model.learn_lambda
            % dL/dlambda
            paramsFor = params; paramsFor(end) = paramsFor(end)+eps;
            llfor = dgpObj(paramsFor,model);
            paramsBack = params; paramsBack(end) = paramsBack(end)-eps;
            llback = dgpObj(paramsBack,model);
            dL_dlambda = -(llfor-llback)/(2*eps);
        end
    else
        eps = 1e-6;
        % dL/dLabmda1
        
        if model.learn_EZ
            inds = K+2*K*R;
        else
            inds = K;
        end
        if model.learn_Lambda1
        dL_dLambda1 = zeros(inpR,1);
        for r = 1:inpR
            paramsFor = params; paramsFor(inds+r) = paramsFor(inds+r)+eps;
            llfor = dgpObj(paramsFor,model);
            paramsBack = params; paramsBack(inds+r) = paramsBack(inds+r)-eps;
            llback = dgpObj(paramsBack,model);
            dL_dLambda1(r) = -(llfor-llback)/(2*eps);
        end
        inds = inds+inpR;
        end
        if model.learn_sigma1
            % dL/dsigma1
            paramsFor = params; paramsFor(inds+1) = paramsFor(inds+1)+eps;
            llfor = dgpObj(paramsFor,model);
            paramsBack = params; paramsBack(inds+1) = paramsBack(inds+1)-eps;
            llback = dgpObj(paramsBack,model);
            dL_dsigma1 = -(llfor-llback)/(2*eps);
            inds = inds+1;
        end
        if model.learn_beta1
            % dL/dbeta1
            paramsFor = params; paramsFor(inds+1) = paramsFor(inds+1)+eps;
            llfor = dgpObj(paramsFor,model);
            paramsBack = params; paramsBack(inds+1) = paramsBack(inds+1)-eps;
            llback = dgpObj(paramsBack,model);
            dL_dbeta1 = -(llfor-llback)/(2*eps);
            inds = inds+1;
        end
        % dL/dLabmda2
        if model.learn_Lambda2
        dL_dLambda2 = zeros(R,1);
        for r = 1:R
            paramsFor = params; paramsFor(inds+r) = paramsFor(inds+r)+eps;
            llfor = dgpObj(paramsFor,model);
            paramsBack = params; paramsBack(inds+r) = paramsBack(inds+r)-eps;
            llback = dgpObj(paramsBack,model);
            dL_dLambda2(r) = -(llfor-llback)/(2*eps);
        end
        inds = inds+R;
        end
        if model.learn_sigma2
            % dL/dsigma2
            inds = K+2*K*R+inpR+2+R;
            paramsFor = params; paramsFor(inds+1) = paramsFor(inds+1)+eps;
            llfor = dgpObj(paramsFor,model);
            paramsBack = params; paramsBack(inds+1) = paramsBack(inds+1)-eps;
            llback = dgpObj(paramsBack,model);
            dL_dsigma2 = -(llfor-llback)/(2*eps);
            inds = inds+1;
        end
        if model.learn_beta2
            % dL/dbeta2
            paramsFor = params; paramsFor(inds+1) = paramsFor(inds+1)+eps;
            llfor = dgpObj(paramsFor,model);
            paramsBack = params; paramsBack(inds+1) = paramsBack(inds+1)-eps;
            llback = dgpObj(paramsBack,model);
            dL_dbeta2 = -(llfor-llback)/(2*eps);
            inds = inds+1;
        end
        if model.learn_lambda
            % dL/dlambda
            paramsFor = params; paramsFor(inds+1) = paramsFor(inds+1)+eps;
            llfor = dgpObj(paramsFor,model);
            paramsBack = params; paramsBack(inds+1) = paramsBack(inds+1)-eps;
            llback = dgpObj(paramsBack,model);
            dL_dlambda = -(llfor-llback)/(2*eps);
        end
        if model.learn_warp
            dL_dwarpx_c = zeros(inpR,1);
            for r = 1:inpR
                paramsFor = params; paramsFor(inds+r) = paramsFor(inds+r)+eps;
                llfor = dgpObj(paramsFor,model);
                paramsBack = params; paramsBack(inds+r) = paramsBack(inds+r)-eps;
                llback = dgpObj(paramsBack,model);
                dL_dwarpx_c(r) = -(llfor-llback)/(2*eps);
            end
            inds = inds+inpR;
            dL_dwarpx_l = zeros(inpR,1);
            for r = 1:inpR
                paramsFor = params; paramsFor(inds+r) = paramsFor(inds+r)+eps;
                llfor = dgpObj(paramsFor,model);
                paramsBack = params; paramsBack(inds+r) = paramsBack(inds+r)-eps;
                llback = dgpObj(paramsBack,model);
                dL_dwarpx_l(r) = -(llfor-llback)/(2*eps);
            end
            inds = inds+inpR;
            dL_dwarpx_s = zeros(inpR,1);
            for r = 1:inpR
                paramsFor = params; paramsFor(inds+r) = paramsFor(inds+r)+eps;
                llfor = dgpObj(paramsFor,model);
                paramsBack = params; paramsBack(inds+r) = paramsBack(inds+r)-eps;
                llback = dgpObj(paramsBack,model);
                dL_dwarpx_s(r) = -(llfor-llback)/(2*eps);
            end
        end
        %fprintf('use approximations\n');
    end
    grad = [];
    if model.learn_u
        grad = [grad;dLM_du+dLG_du];%for dL/du
    end
    if model.learn_EZ
        grad = [grad;dLDBV_dE+dLM_dE;%for dL/dE
            dLM_dZ+dLG_dZ]; %for dL/dZ
    end
    if model.learn_Lambda1
    grad = [grad;dL_dLambda1];
    end
    if model.learn_sigma1
        grad = [grad;dL_dsigma1];
    end
    if model.learn_beta1
        grad = [grad;dL_dbeta1];
    end
    if model.learn_Lambda2
    grad = [grad;dL_dLambda2];
    end
    if model.learn_sigma2
        grad = [grad;dL_dsigma2];
    end
    if model.learn_beta2
        grad = [grad;dL_dbeta2];
    end
    if model.learn_lambda
        grad = [grad;dL_dlambda];
    end
    if model.learn_warp
        grad = [grad;dL_dwarpx_c;dL_dwarpx_l;dL_dwarpx_s];
    end
    grad = -grad;
    
end