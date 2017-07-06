function gradBeta = ardjitkernelGradBeta(X,Y,Lambda,beta,sigma)
% TESTED!
iw_sqrt = sqrt(Lambda);
if ~isempty(Y)
    X_scaled = bsxfun(@times,iw_sqrt,X);
    Y_scaled = bsxfun(@times,iw_sqrt,Y);
    d_uf = bsxfun(@plus,sum(X_scaled.^2,2),sum(Y_scaled.^2,2)') - 2*(X_scaled*(Y_scaled'));
    d_uf = max(d_uf,0);
    nconst = exp(-0.5*sigma*sum(Lambda));
    gradBeta = nconst*exp(-0.5*d_uf); % Noise is uniform.
else
    X_scaled = bsxfun(@times,iw_sqrt,X);
    d_uu = bsxfun(@plus,sum(X_scaled.^2,2),sum(X_scaled.^2,2)') - 2*(X_scaled*(X_scaled'));
    d_uu = max(d_uu,0);
    nconst = exp(-0.5*sigma*sum(Lambda));
    nmat = nconst*ones(size(X,1)) + (1-nconst)*eye(size(X,1));
    gradBeta = exp(-0.5*d_uu).*nmat; 
end
