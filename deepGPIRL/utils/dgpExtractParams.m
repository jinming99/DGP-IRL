function params = dgpExtractParams(model)
% params follow the order of:
% variational params: u:1*K,E:1*(K*R) (e1,e2,...,eR)
% inducing inputs: Z: 1*(K*R)
% hyper param: Lambda1: 1*inpR, sigma1^2:1*1,beta1: 1*1, 
% Lambda2: 1*R, sigma2^2: 1*1, beta2: 1*1, lambda: 1*1
% warp_c, warp_l, warp_s of size inpR*3

%% first extract and update parameters
K = model.K; R = model.R; N = model.N; inpR = model.inpR;
ids = 0; ide = 0;
if model.learn_u
    ids = 1; ide = K; params(ids:ide) = model.u;
end
if model.learn_EZ
ids = ide+1; ide = ide+K*R; params(ids:ide) = model.E(:);
ids = ide+1; ide = ide+K*R; params(ids:ide) = model.Z(:);
end
if model.learn_Lambda1
ids = ide+1; ide = ide+inpR; params(ids:ide) = gpirlhpxform(model.Lambda1,[],model.ard_xform,3); % This is \Lambda1
end
if model.learn_sigma1
    ids = ide+1; ide = ide+1; params(ids:ide) = gpirlhpxform(model.sigma1,[],model.ard_xform,3);
end
if model.learn_beta1
    ids = ide+1; ide = ide+1; params(ids:ide) = gpirlhpxform(model.beta1,[],model.ard_xform,3);
end
if model.learn_Lambda2
ids = ide+1; ide = ide+R; params(ids:ide) = gpirlhpxform(model.Lambda2,[],model.ard_xform,3);
end
if model.learn_sigma2
    ids = ide+1; ide = ide+1; params(ids:ide) = gpirlhpxform(model.sigma2,[],model.ard_xform,3);
end
if model.learn_beta2
    ids = ide+1; ide = ide+1; params(ids:ide) = gpirlhpxform(model.beta2,[],model.ard_xform,3);
end
if model.learn_lambda
    ids = ide+1; ide = ide+1; params(ids:ide) = gpirlhpxform(model.lambda,[],model.ard_xform,3);
end
if model.learn_warp
    ids = ide+1; ide = ide+inpR; 
    params(ids:ide) = gpirlhpxform(model.warp_c,[],model.warp_c_xform,3); % This is m
    ids = ide+1; ide = ide+inpR; 
    params(ids:ide) = gpirlhpxform(model.warp_l,[],model.warp_l_xform,3); % This is \ell
    ids = ide+1; ide = ide+inpR; 
    params(ids:ide) = gpirlhpxform(model.warp_s,[],model.warp_s_xform,3); % This is s    
end
if size(params,1)<size(params,2)
    params = params';
end