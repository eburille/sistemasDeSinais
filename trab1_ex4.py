function [dy_dt] = ode_func2(t,y,a,b)
    
    y1 = y(1)
    z = y(2)
    dy1_dt = z;
    dy2_dt = -(a)*z - (b)*y1+cos(0.5.*t)+sin(0.5.*t);
    dy_dt = [dy1_dt; dy2_dt];
end
import numpy as np

def f(t, y, a, b):
    y1 = y(1)
    z = y(2)

    dy1 = z
    dy2 = -a*z - b*y1 + np.cos(0.5*t) + np.sin(0.5*t)
    dy_dt = [dy1, dy2]
    return dy_dt
