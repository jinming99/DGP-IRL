function SigmaB = calc_SigmaB(X,W,Lambda,sigma,beta)
KXX = ardjitkernel(X,[],Lambda,beta,sigma);
KWW = ardjitkernel(W,[],Lambda,beta,sigma);
KXW = ardjitkernel(X,W,Lambda,beta,sigma);
[~,~,invKWW] = gpirlsafeinv(KWW,zeros(size(W,1),1));
SigmaB = (KXX-KXW*invKWW*KXW').*eye(size(X,1));
