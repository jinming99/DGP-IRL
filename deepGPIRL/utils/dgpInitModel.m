function model = dgpInitModel(model)
%% Model initialization!
% Get the hyperparameters right
model.Lambda1 = 1./((max(model.X)-min(model.X)).^2)*model.Lambda_init;%previous 10
model.beta1 = model.beta_init*max(var(model.X));
%model.beta1 = model.beta_init*var(model.X(:));
model.sigma1 = (1e-3/sum(model.Lambda1))*model.sigma_init;%1e-3/sum(model.Lambda1);
% Guess the initial point for D
model.D = ppcaEmbed(model.X, model.R);
%model.D = model.X;
model.D = bsxfun(@rdivide,model.D,var(model.D));
model.Lambda2 = 1./((max(model.D)-min(model.D)).^2)*model.Lambda_init;%10
model.beta2 = model.beta_init*max(var(model.D));%
model.sigma2 = (1e-3/sum(model.Lambda2))*model.sigma_init;%1e-3/sum(model.Lambda2);

% Guess the initial point for E
model.KXW = ardjitkernel(model.X,model.W,model.Lambda1,model.beta1,model.sigma1);
model.KWW = ardjitkernel(model.W,[],model.Lambda1,model.beta1,model.sigma1);
model.E = model.KWW*pinv(model.KXW)*model.D;
% randomly initialize Z
model.Z = var(model.D(:))*rand(model.K,model.R);
% noise variance lambda
model.lambda = model.SNR_init/var(model.D(:));
model.lambda_min = 10/var(model.D(:));
model.lambda_max = 1e5/var(model.D(:));
if model.warp_x
    model.warp_c = model.warp_c_init*ones(1,model.inpR);
    model.warp_l = model.warp_l_init*ones(1,model.inpR);
    model.warp_s = model.warp_s_init*ones(1,model.inpR);
end
