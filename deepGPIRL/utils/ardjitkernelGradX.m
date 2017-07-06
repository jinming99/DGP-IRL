function gradX = ardjitkernelGradX(X,Y,Lambda,beta,sigma,spec)
iw_sqrt = sqrt(Lambda);
N = size(X,1); R = size(X,2);
if ~isempty(Y)&&spec == 1
    K = size(Y,1);
    X_scaled = bsxfun(@times,iw_sqrt,X);
    Y_scaled = bsxfun(@times,iw_sqrt,Y);
    d_uf = bsxfun(@plus,sum(X_scaled.^2,2),sum(Y_scaled.^2,2)') - 2*(X_scaled*(Y_scaled'));
    d_uf = max(d_uf,0);
    nconst = exp(-0.5*sigma*sum(Lambda));
    KXY = nconst*beta*exp(-0.5*d_uf); % Noise is uniform.
    
    %gradX = zeros(N*R,N*K);
    Xt = repmat(X,K,1);
    Yt = kron(Y,ones(N,1));
    vv = -repmat(KXY(:)',1,R).*(Xt(:)'-Yt(:)').*kron(Lambda,ones(1,K*N));
    idx = kron((0:(R-1))*N,ones(1,K*N))+repmat(1:N,1,K*R);
    idy = repmat(kron((0:(K-1))*N,ones(1,N)),1,R)+repmat(1:N,1,K*R);
    gradX = sparse(idx,idy,vv,N*R,N*K);
elseif ~isempty(Y)&&spec == 2
    K = size(Y,1);
    X_scaled = bsxfun(@times,iw_sqrt,X);
    Y_scaled = bsxfun(@times,iw_sqrt,Y);
    d_uf = bsxfun(@plus,sum(X_scaled.^2,2),sum(Y_scaled.^2,2)') - 2*(X_scaled*(Y_scaled'));
    d_uf = max(d_uf,0);
    nconst = exp(-0.5*sigma*sum(Lambda));
    KXY = nconst*beta*exp(-0.5*d_uf); % Noise is uniform.
    
    Xt = repmat(X,K,1);
    Yt = kron(Y,ones(N,1));
    vv = -repmat(KXY(:)',1,R).*(Yt(:)'-Xt(:)').*kron(Lambda,ones(1,K*N));
    idx = repmat(kron(1:K,ones(1,N)),1,R) + kron((0:(R-1))*K,ones(1,K*N));
    idy = repmat(1:N,1,K*R)+repmat(kron((0:(K-1))*N,ones(1,N)),1,R);
    gradX = sparse(idx,idy,vv,K*R,N*K);
    
%    
%     
%     for r = 1:R
%         for j = 1:K
%             for i = 1:N
%                 gradX(j+(r-1)*K,i+(j-1)*N) = -(Y(j,r)-X(i,r))*Lambda(r)*KXY(i,j);
%             end
%         end
%     end
            
    
else
    X_scaled = bsxfun(@times,iw_sqrt,X);
    d_uu = bsxfun(@plus,sum(X_scaled.^2,2),sum(X_scaled.^2,2)') - 2*(X_scaled*(X_scaled'));
    d_uu = max(d_uu,0);
    nconst = exp(-0.5*sigma*sum(Lambda));
    nmat = nconst*ones(size(X,1)) + (1-nconst)*eye(size(X,1));
    KXX = beta*exp(-0.5*d_uu).*nmat;
    
    idx1 = repmat(1:N,1,R*N)+kron((0:(R-1))*N,ones(1,N*N));
    idy1 = repmat(1:N,1,R*N)+repmat(kron((0:(N-1))*N,ones(1,N)),1,R);
    Xt11 = repmat(X,N,1);
    Xt12 = kron(X,ones(N,1));
    vv1 = -kron(Lambda,ones(1,N*N)).*(Xt11(:)'-Xt12(:)').*repmat(KXX(:)',1,R);
%     for r = 1:R
%         for l = 1:N
%             for i = 1:N
%                 gradX(i+(r-1)*N,i+(l-1)*N) = -(X(i,r)-X(l,r))*Lambda(r)*KXX(i,l);
%             end
%         end
%     end
%     

    idx2 = repmat(kron(1:N,ones(1,N)),1,R) + kron((0:(R-1))*N,ones(1,N*N));
    idy2 = repmat(1:N,1,R*N) + repmat(kron((0:(N-1))*N,ones(1,N)),1,R);
    Xt21 = kron(X,ones(N,1));
    Xt22 = repmat(X,N,1);
    vv2 = -kron(Lambda,ones(1,N*N)).*(Xt21(:)'-Xt22(:)').*repmat(KXX(:)',1,R);
    gradX = sparse([idx1,idx2],[idy1,idy2],[vv1,vv2],N*R,N*N);
%     for r = 1:R
%         for i = 1:N
%             for k = 1:N
%                 gradX(i+(r-1)*N,k+(i-1)*N) = -(X(i,r)-X(k,r))*Lambda(r)*KXX(k,i);
%             end
%         end
%     end
%     
%     for k = 1:N
%         for l = 1:N
%             for i = 1:N
%                 for r = 1:R
%                     if k == i
%                         gradX(i+(r-1)*N,i+(l-1)*N) = -(X(i,r)-X(l,r))*Lambda(r)*KXX(i,l);
%                     elseif l == i
%                         gradX(i+(r-1)*N,k+(i-1)*N) = -(X(i,r)-X(k,r))*Lambda(r)*KXX(k,i);
%                     end
%                 end
%             end
%         end
%     end
                  
end
