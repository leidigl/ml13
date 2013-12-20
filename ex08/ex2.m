close all;

X = [-0.8372, -0.4558, 0.6902, 0.1114, -0.4678];
Y = [-1.1414, -1.5286, -1.1893, -1.9021, -1.5595];

Xstar = [-0.5, 0.5];
YstarMean = zeros(1,2);

%% a)

l = 1;

% create K
K = zeros(size(X,2), size(X,2));
for i = 1:size(X,2)
    for j = 1:size(X,2)
        K(i,j) = kernel(X(i),X(j), l);
    end
end

for x = 1:size(Xstar,2)

    Kstar = zeros(size(X,2), 1);
    for i = 1:size(X,2)
        Kstar(i,1) = kernel(X(i), Xstar(x), l);
    end

    KstarStar = kernel(Xstar(x), Xstar(x), l);

    f = zeros(size(X,2),1);
    for i = 1:size(X,2)
        f(i) = normpdf(X(i), 0, kernel(X(i), X(i), l));
    end

    YstarMean(x) = Kstar'*inv(K)* f;

    sigmaStar = KstarStar - Kstar'*inv(K)*Kstar;
end

disp(YstarMean);

%% b)
for l = 0.01:0.05:0.5
    disp(l);
    h = figure;
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
            f(i) = normpdf(X(i), 0, kernel(X(i), X(i), l));
        end

        YstarMean(x) = Kstar'*inv(K)* f;
        sigmaStar = KstarStar - Kstar'*inv(K)*Kstar;
        
        hold on;
        subplot(2,1,x);
        plot(-1:0.001:1,normpdf(-1:0.001:1,YstarMean(x),sigmaStar));
        title([num2str(l, 'l = %.2f') ', ' num2str(Xstar(x), 'x* = %.2f')]);
        
    end
    o = num2str(l, 'plots/l%.2f.svg');
    plot2svg(o);
end

