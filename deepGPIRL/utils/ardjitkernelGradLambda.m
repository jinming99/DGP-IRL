function gradLambda = ardjitkernelGradLambda(X,Y,Lambda,beta,sigma)
% TESTED!
iw_sqrt = sqrt(Lambda); R = size(X,2);
N = size(X,1);
if ~isempty(Y)
    K = size(Y,1);
    X_scaled = bsxfun(@times,iw_sqrt,X);
    Y_scaled = bsxfun(@times,iw_sqrt,Y);
    d_uf = bsxfun(@plus,sum(X_scaled.^2,2),sum(Y_scaled.^2,2)') - 2*(X_scaled*(Y_scaled'));
    d_uf = max(d_uf,0);
    nconst = exp(-0.5*sigma*sum(Lambda));
    KXY = nconst*beta*exp(-0.5*d_uf); % Noise is uniform.
    gradLambda = zeros(N,K,R);
    for r = 1:R
        T_mat = bsxfun(@minus,X(:,r),Y(:,r)');
        T_mat = -0.5*T_mat.^2-0.5*sigma*ones(N,K);
        gradLambda(:,:,r) = T_mat.*KXY; 
    end
else
    X_scaled = bsxfun(@times,iw_sqrt,X);
    d_uu = bsxfun(@plus,sum(X_scaled.^2,2),sum(X_scaled.^2,2)') - 2*(X_scaled*(X_scaled'));
    d_uu = max(d_uu,0);
    nconst = exp(-0.5*sigma*sum(Lambda));
    nmat = nconst*ones(size(X,1)) + (1-nconst)*eye(size(X,1));
    KXX = beta*exp(-0.5*d_uu).*nmat; 
    gradLambda = zeros(N,N,R);
    for r = 1:R
        T_mat = bsxfun(@minus,X(:,r),X(:,r)');
        T_mat = -0.5*T_mat.^2-0.5*sigma*(ones(N)-eye(N));
        gradLambda(:,:,r) = T_mat.*KXX; 
    end
end
