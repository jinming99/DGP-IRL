% Optimized kernel computation function for DC mode GPIRL.
function KDstarZ = dgpirlkernel(model,Xstar)

%% first extract and update parameters
K = model.K; R = model.R; N = model.N; inpR = model.inpR;
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
%% Now calculate essential statistics
% Compute scales.
iw_sqrt1 = sqrt(Lambda1);
iw_sqrt2 = sqrt(Lambda2);

%% Layer 1
X = Xstar;
W = model.W;
X_scaled = bsxfun(@times,iw_sqrt1,X);
W_scaled = bsxfun(@times,iw_sqrt1,W);
% Noise is uniform.
nconst = exp(-0.5*sigma1*sum(Lambda1));
nmat = nconst*ones(K) + (1-nconst)*eye(K);


% Compute K_WX matrix.
d_uf = bsxfun(@plus,sum(X_scaled.^2,2),sum(W_scaled.^2,2)') - 2*(X_scaled*(W_scaled'));
d_uf = max(d_uf,0);
KXstarW = nconst*beta1*exp(-0.5*d_uf); % Noise is uniform.

%% Layer 2
Z = model.Z;
Dstar = KXstarW*model.invKWW*model.E;
D_scaled = bsxfun(@times,iw_sqrt2,Dstar);
Z_scaled = bsxfun(@times,iw_sqrt2,Z);
% Noise is uniform.
nconst = exp(-0.5*sigma2*sum(Lambda2));
nmat = nconst*ones(K) + (1-nconst)*eye(K);

% Compute KDstarZ matrix.
d_uf = bsxfun(@plus,sum(D_scaled.^2,2),sum(Z_scaled.^2,2)') - 2*(D_scaled*(Z_scaled'));
d_uf = max(d_uf,0);
KDstarZ = nconst*beta2*exp(-0.5*d_uf); % Noise is uniform.
