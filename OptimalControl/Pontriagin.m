load('trajectory.mat')
T=200;

solinit = bvpinit(linspace(0,T,1500),[0 0 0 0]);
sol = bvp4c(@fourode,@(ya,yb)fourbc(ya,yb,trajectory(1,:)),solinit);


%%
x_val=sol.y(1,:);
v_val=sol.y(2,:);
t_val=sol.x;
Q = trapz(t_val,x_val.*x_val);
Q_RL = sumsqr(trajectory(:,1));
control_check=max(abs(diff(trajectory(:,2))));
plot(t_val,x_val)
hold on
plot(trajectory(:,1))
%%

function dydx = fourode(x,y)
dydx = [y(2); -0.0001*atan(y(4)*8)*2/pi;-2*y(1);-y(3)];
end

function res = fourbc(ya,yb,stp)
res = [ya(1)-stp(1);ya(2)-stp(2);yb(3);yb(4)];
end
