function model_n = dgpUpdateStats(model,params)
% params follow the order of:
% variational params: u:1*K,E:1*(K*R) (e1,e2,...,eR),
% inducing inputs: Z: 1*(K*R)
% hyper param: Lambda1: 1*inpR, sigma1^2:1*1,beta1: 1*1, 
% Lambda2: 1*R, sigma2^2: 1*1, beta2: 1*1, lambda: 1*1

%% first extract and update parameters
if size(params,1)<size(params,2)
    params = params';
end
K = model.K; R = model.R; N = model.N; inpR = model.inpR;
ids = 0; ide = 0;
if model.learn_u
    ids = 1; ide = K; model.u = params(ids:ide);
end
if model.learn_EZ
    ids = ide+1; ide = ide+K*R; model.E = reshape(params(ids:ide),K,R);
    ids = ide+1; ide = ide+K*R; model.Z = reshape(params(ids:ide),K,R);
end
if model.learn_Lambda1
ids = ide+1; ide = ide+inpR; model.Lambda1 = gpirlhpxform(params(ids:ide)',[],model.ard_xform,1); % This is \Lambda1
end
if model.learn_sigma1
    ids = ide+1; ide = ide+1; model.sigma1 = gpirlhpxform(params(ids:ide),[],model.ard_xform,1);
end
if model.learn_beta1
    ids = ide+1; ide = ide+1; model.beta1 = gpirlhpxform(params(ids:ide),[],model.ard_xform,1);
end
if model.learn_Lambda2
ids = ide+1; ide = ide+R; model.Lambda2 = gpirlhpxform(params(ids:ide)',[],model.ard_xform,1);
end
if model.learn_sigma2
    ids = ide+1; ide = ide+1; model.sigma2 = gpirlhpxform(params(ids:ide),[],model.ard_xform,1);
end
if model.learn_beta2
    ids = ide+1; ide = ide+1; model.beta2 = gpirlhpxform(params(ids:ide),[],model.ard_xform,1);
end
if model.learn_lambda
    ids = ide+1; ide = ide+1; model.lambda = gpirlhpxform(params(ids:ide),[],model.ard_xform,1);
    if model.lambda > model.lambda_max model.lambda = model.lambda_max;
    elseif model.lambda < model.lambda_min model.lambda = model.lambda_min;end
end
if model.learn_warp
    ids = ide+1; ide = ide+inpR; 
    model.warp_c = gpirlhpxform(params(ids:ide),[],model.warp_c_xform,1)'; % This is m
    ids = ide+1; ide = ide+inpR; 
    model.warp_l = gpirlhpxform(params(ids:ide),[],model.warp_l_xform,1)'; % This is \ell
    ids = ide+1; ide = ide+inpR; 
    model.warp_s = gpirlhpxform(params(ids:ide),[],model.warp_s_xform,1)'; % This is s    
end

%% Extract out for convenience
Lambda1 = model.Lambda1; % This is \Lambda1
sigma1 = model.sigma1; % This is \sigma_1^2
beta1 = model.beta1; % This is \beta1
Lambda2 = model.Lambda2; % This is \Lambda2
sigma2 = model.sigma2; % This is \sigma_2^2
beta2 = model.beta2; % This is \beta2
lambda = model.lambda;
Lambda1 = min(Lambda1,1e100); % Prevent overflow.
Lambda2 = min(Lambda2,1e100); % Prevent overflow.
if model.warp_x
    warp_c = model.warp_c;
    warp_l = model.warp_l;
    warp_s = model.warp_s;
end
%% Now calculate essential statistics
% Compute scales.
iw_sqrt1 = sqrt(Lambda1);
iw_sqrt2 = sqrt(Lambda2);

%% Layer 1
X = model.X;
W = model.W;

% Scale positions in feature space.
if model.warp_x,
    [W_warped,dW] = gpirlwarpx(W,warp_c,warp_l,warp_s);
    [X_warped,dX] = gpirlwarpx(X,warp_c,warp_l,warp_s);
else
    W_warped = W;
    X_warped = X;
end;
X_scaled = bsxfun(@times,iw_sqrt1,X_warped);
W_scaled = bsxfun(@times,iw_sqrt1,W_warped);

% Noise is uniform, construct noise matrix.
mask_mat = ones(K)-eye(K);
if model.warp_x,
    % Noise is spatially varying.
    dxu_scaled = -0.25*sigma1*bsxfun(@times,Lambda1,dW);
    dxu_ssum = sum(dxu_scaled,2);
    nudist = bsxfun(@plus,dxu_ssum,dxu_ssum');
    nudist(~mask_mat) = 0;
    nmat = exp(nudist);
else
    % Noise is uniform.
    nconst = exp(-0.5*sigma1*sum(Lambda1));
    nmat = nconst*ones(K) + (1-nconst)*eye(K);
end;

% Compute K_WW matrix.
d_uu = bsxfun(@plus,sum(W_scaled.^2,2),sum(W_scaled.^2,2)') - 2*(W_scaled*(W_scaled'));
d_uu = max(d_uu,0);
K_WW = beta1*exp(-0.5*d_uu).*nmat; 
model.KWW = K_WW;

% Compute K_uf matrix.
d_uf = bsxfun(@plus,sum(W_scaled.^2,2),sum(X_scaled.^2,2)') - 2*(W_scaled*(X_scaled'));
d_uf = max(d_uf,0);
if model.warp_x,
    % Noise is spatially varying.
    dxf_scaled = -0.25*beta1*bsxfun(@times,Lambda1,dX);
    dxf_ssum = sum(dxf_scaled,2);
    nfdist = bsxfun(@plus,dxu_ssum,dxf_ssum');
    K_WX = beta1*exp(-0.5*d_uf).*exp(nfdist);
else
    % Noise is uniform.
    K_WX = nconst*beta1*exp(-0.5*d_uf);
end;
model.KWX = K_WX;

% Compute K_XX matrix.
mask_mat = ones(N)-eye(N);
if model.warp_x,
    % Noise is spatially varying.
    dxf_scaled = -0.25*sigma1*bsxfun(@times,Lambda1,dX);
    dxf_ssum = sum(dxf_scaled,2);
    nudist = bsxfun(@plus,dxf_ssum,dxf_ssum');
    nudist(~mask_mat) = 0;
    nmat = exp(nudist);
else
    % Noise is uniform.
    nconst = exp(-0.5*sigma1*sum(Lambda1));
    nmat = nconst*ones(N) + (1-nconst)*eye(N);
end;

% Compute K_XX matrix.
d_ff = bsxfun(@plus,sum(X_scaled.^2,2),sum(X_scaled.^2,2)') - 2*(X_scaled*(X_scaled'));
d_ff = max(d_ff,0);
K_XX = beta1*exp(-0.5*d_ff).*nmat; 
model.KXX = K_XX;

% Invert the kernel matrix KWW
try
    [model.invKWW_E,model.halfLogDetKWW,invKWW] = gpirlsafeinv(K_WW,model.E);
    model.invKWW = invKWW;
catch err
    save dump_file_chol; % Save dump.
    rethrow(err);% Display the error.
end;
model.KXW_invKWW = K_WX'*invKWW; 
model.KKKK_WX = model.KXW_invKWW'*model.KXW_invKWW;
model.KWX_KXW_invKWW = model.KWX*model.KXW_invKWW;
E2sum = zeros(K,K);
for i = 1:R
    E2sum = E2sum + model.E(:,i)*model.E(:,i)';
end
model.E2sum = E2sum;

try
    [~,model.halfLogDetGamma,model.Gamma] = gpirlsafeinv(1/lambda*model.KWW+model.KWX*model.KWX',zeros(K,1));
catch err
    save dump_file_chol; % Save dump.
    rethrow(err);% Display the error.
end;
model.Sigmav = 1/lambda*model.KWW*model.Gamma*model.KWW;
model.halfLogDetSigmav = model.halfLogDetGamma-0.5*K*log(lambda)+2*model.halfLogDetKWW;
model.Gamma_KWX = model.Gamma*model.KWX;
model.V = model.KWW*model.Gamma_KWX*model.KXW_invKWW*model.E;

model.VVT = zeros(K,K,R);
for r = 1:R
    model.VVT(:,:,r) = model.V(:,r)*model.V(:,r)';
end
model.VVTsum = sum(model.VVT,3);

%% Layer 2
Z = model.Z;
D = model.KXW_invKWW*model.E;
model.D = D;
VDsum = zeros(K,N);
D2sum = zeros(N,N);
for r = 1:R
    VDsum = VDsum+model.V(:,r)*model.D(:,r)';
    D2sum = D2sum+model.D(:,r)*model.D(:,r)';
end
model.VDsum = VDsum;
model.D2sum = D2sum;
D_scaled = bsxfun(@times,iw_sqrt2,D);
Z_scaled = bsxfun(@times,iw_sqrt2,Z);
% Noise is uniform.
nconst = exp(-0.5*sigma2*sum(Lambda2));
nmat = nconst*ones(K) + (1-nconst)*eye(K);

% Compute K_ZZ matrix.
d_uu = bsxfun(@plus,sum(Z_scaled.^2,2),sum(Z_scaled.^2,2)') - 2*(Z_scaled*(Z_scaled'));
d_uu = max(d_uu,0);
K_ZZ = beta2*exp(-0.5*d_uu).*nmat; 
model.KZZ = K_ZZ;

% Compute K_ZD matrix.
d_uf = bsxfun(@plus,sum(Z_scaled.^2,2),sum(D_scaled.^2,2)') - 2*(Z_scaled*(D_scaled'));
d_uf = max(d_uf,0);
K_ZD = nconst*beta2*exp(-0.5*d_uf); % Noise is uniform.
model.KZD = K_ZD;

% Invert the kernel matrix KZZ
try
    [model.invKZZ_u,model.halfLogDetKZZ,invKZZ] = gpirlsafeinv(K_ZZ,model.u);
catch err
    save dump_file_chol; % Save dump.
    rethrow(err);% Display the error.
end;
model.invKZZ = invKZZ;
model.KDZ_invKZZ = K_ZD'*invKZZ; 
model.SigmaB = (model.KXX-model.KXW_invKWW*model.KWX).*eye(N);

model_n = model;