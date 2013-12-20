l = 1;

X = [-0.8372, -0.4558, 0.6902, 0.1114, -0.4678];
Y = [-1.1414, -1.5286, -1.1893, -1.9021, -1.5595];

Xstar = [-0.5, 0.5];
YstarMean = zeros(1,2);

% create K
K = zeros(size(X,2), size(X,2));
for i = 1:size(X,2)
    for j = 1:size(X,2)
        K(i,j) = kernel(X(i),X(j), l);
    end
end

% compute the mean value ystar for each xstar
for x = 1:size(Xstar,2)
    
    Kstar = zeros(size(X,2), 1);
    for i = 1:size(X,2)
        Kstar(i,1) = kernel(X(i), Xstar(x), l);
    end
    
    KstarStar = kernel(Xstar(x), Xstar(x), l);
    
    f = zeros(size(X,2),1);
    for i = 1:size(X,2)
        f(i) = normpdf(X(i), 0, kernel(X(i), X(i)));
    end
    
    YstarMean(x) = Kstar'*inv(K)* f;
end

