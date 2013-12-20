function [ w ] = kernel( x1, x2, l )

    sigmaF = 1;
    sigmaN = 0.5;
    delta = 0;  % no noise
    
    w = sigmaF^2 * exp(-1/(2*l^2)*(x1-x2)^2) + sigmaN^2*delta;

end

