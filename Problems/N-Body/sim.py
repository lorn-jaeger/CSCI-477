import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation,rc
from IPython.display import HTML, display, clear_output

#################################################
# Simulation Functions
#################################################

def EulerRichardson(dt, f, t, y, *args):
    k1 = f(t, y, *args)
    y_mid = y + k1 * (dt/2)
    k2 = f(t + dt/2, y_mid, *args)
    return y + k2 * dt

def RungeKutta4(dt, f, t, y, *args):
    k1 = f(t, y, *args)
    k2 = f(t + dt / 2, y + (dt / 2) * k1, *args)
    k3 = f(t + dt / 2, y + (dt / 2) * k2, *args)
    k4 = f(t + dt, y + dt * k3, *args)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def RungeKutta45(dt, f, t, y, *args):
    c2, c3, c4, c5, c6, c6 = 1/5, 3/10, 4/5, 8/9, 1, 1
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    a71, a72, a73, a74, a75, a76 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
    b1, b2, b3, b4, b5, b6, b7 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0
    bs1, bs2, bs3, bs4, bs5, bs6, bs7 = 5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40

    k1 = f(t, y, *args)
    k2 = f(t + c2 * dt, y + dt * (a21 * k1), *args)
    k3 = f(t + c3 * dt, y + dt * (a31 * k1 + a32 * k2), *args)
    k4 = f(t + c4 * dt, y + dt * (a41 * k1 + a42 * k2 + a43 * k3), *args)
    k5 = f(t + c5 * dt, y + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), *args)
    k6 = f(t + c6 * dt, y + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), *args)
    k7 = f(t + dt, y + dt * (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6), *args)

    y_new = y + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
    ys_new = y + dt * (bs1 * k1 + bs2 * k2 + bs3 * k3 + bs4 * k4 + bs5 * k5 + bs6 * k6 + bs7 * k7)

    err = np.abs(y_new - ys_new)

    return y_new, err

def solve_ode(f, tspan, y0, method=RungeKutta45, *args, **options):
   t0, tf = tspan
    dt = options.get('first_step', 0.01)
    atol = options.get('atol', 1e-6)
    rtol = options.get('rtol', 1e-6)
    adaptive = method in [RungeKutta45]

    t = [t0]
    y = [y0]

    while t[-1] < tf:
        y_c = y[-1]
        t_c = t[-1]

        if adaptive:
            y_new, err = method(dt, f, t_c, y_c, *args)
        else:
            y_new = method(dt, f, t_c, y_c, *args)
            err = np.zeros_like(y_c)

        if adaptive:
            scale = atol + rtol * np.abs(y_c)

            scaled_err = np.sqrt(np.mean((err / scale) ** 2))

            if scaled_err < 1.0:
                t.append(t_c + dt)
                y.append(y_new)

                dt = dt * min(2.0, 0.9 * (1.0 / scaled_err) ** 0.2)
            else:
                dt = dt * max(0.5, 0.9 * (1.0 / scaled_err) ** 0.2)
        else:
            t.append(t_c + dt)
            y.append(y_new)

        if t[-1] + dt > tf:
            dt = tf - t[-1]

    return np.array(t), np.array(y)

def n_body(t, y, p):
  dim, m = p['dimension'], p['m']
  G, fix = p.get('G', 1.0), p.get('fix_first', False)
  n = len(m)
  pos = y[:n * dim].reshape(n, dim)
  acc = np.zeros_like(pos)

  for i in range(n):
    for j in range(i + 1, n):
      r = pos[i] - pos[j]
      r_norm = np.linalg.norm(r)
      if r_norm > 0:
        acc[i] -= G * m[j] * r / r_norm ** 3
        acc[j] += G * m[i] * r / r_norm ** 3

  if fix: acc[0] *= 0
  return np.concatenate([y[n * dim:], acc.ravel()])




# Function to compute the total energy given state array (state vector for a sequence of times.)
def total_energy(y,p):
    """
    INPUTS:
    y - the output of the ODE solver
    p - the parameters dictionary
    OUTPUT:
    the total energy at each time y is provided.
    """
    steps,dofs = y.shape
    d = p['dimension']
    half = dofs // 2

    KE=V=np.zeros(steps)
    # This loop determines total potential energy, works on the
    # first half of data in y, the positions.
    for i in range(0,half,d):
        ri = y[:,i:i+d]
        for j in range(i+d,half,d):
            rj = y[:,j:j+d]
            rij  = np.linalg.norm(ri - rj ,axis=1)
            V += - p['G'] * p['m'][i//d] * p['m'][j//d] / rij
    # This loop determines kinetic energy for each body, works on later half
    # of y, or the velocities.
    for i in range(half,dofs,d):
        KE += 0.5 * p['m'][(i-half)//d] * np.sum(y[:,i:i+d]**2,1)

    return KE + V


#####################################################
# Plotting Functions
#####################################################

def show_anim(t_s,y, dt,y0,trace_length=20,out_time=.05):
    plt.style.use('dark_background')
    d = 2
    c=['tab:red','tab:olive','tab:pink','tab:cyan','tab:purple']
    body_list = []
    trace_list = []

    K = int(out_time/dt)
    t_sd = t_s[::K]
    yd   = y[::K,:]

    from matplotlib.figure import Figure
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    x_min,x_max,y_min,y_max = 1e9,-1e9,1e9,-1e9
    for i in range(0,y0.size//d,d):
        x_t = yd[:,i]
        y_t = yd[:,i+1]
        if x_min > x_t.min(): x_min = x_t.min()
        if x_max < x_t.max(): x_max = x_t.max()
        if y_min > y_t.min(): y_min = y_t.min()
        if y_max < y_t.max(): y_max = y_t.max()

        ph, =  ax.plot(x_t,y_t,'-',color=[.7,.7,.7],linewidth=.7);

    plt.xlim([1.2*x_min,1.2*x_max])
    plt.ylim([1.2*y_min,1.2*y_max])

    ax.axis('off')

    for i in range(0,y0.size//d,d):
        ph, =  ax.plot(y0[i],y0[i+1],'o',color=c[i//d]);
        body_list.append( ph )
        ph, = ax.plot([],[],'-',color=c[i//d])
        trace_list.append( ph )

    def animate(i):
        i = i % (t_sd.size-1)
        for im,j in zip(body_list,range(0,d*len(body_list),d)):
            im.set_xdata( [yd[i+1,j]] )
            im.set_ydata( [yd[i+1,j+1]] )

        if i>trace_length:
            for im,j in zip(trace_list,range(0,d*len(trace_list),d)):
                im.set_xdata( yd[i-trace_length:i+1,j] )
                im.set_ydata( yd[i-trace_length:i+1,j+1] )
        return im

    anim = animation.FuncAnimation(fig, animate, interval=20,frames=t_sd.size-1)
    plt.close(fig)
    clear_output(wait=True)
    return anim


#################################################
# DATA
#################################################

from numpy import array,float64,float128

euler = np.array([0,0,1,0,-1,0,0,0,0,.8,0,-.8])

montgomery = np.array([0.97000436,-0.24308753,-0.97000436,0.24308753, 0., 0.,\
0.466203685, 0.43236573, 0.466203685, 0.43236573,\
-0.93240737,-0.86473146])

lagrange = np.array([1.,0.,-0.5,0.866025403784439, -0.5,-0.866025403784439,\
0.,0.8,-0.692820323027551,-0.4, 0.692820323027551, -0.4])

skinny_pinapple = np.array([0.419698802831,1.190466261252,\
0.076399621771, 0.296331688995,\
0.100310663856, -0.729358656127,\
0.102294566003, 0.687248445943,\
0.148950262064, 0.240179781043,\
-0.251244828060, -0.927428226977])

hand_in_hand_oval = np.array([0.906009977921, 0.347143444587,\
-0.263245299491, 0.140120037700,\
-0.252150695248, -0.661320078799,\
0.242474965162, 1.045019736387,\
-0.360704684300, -0.807167979922,\
0.118229719138, -0.237851756465])

four_body = np.array([1.382857,0,\
0,0.157030,\
-1.382857,0,\
0,-0.157030,\
0,0.584873,\
1.871935,0,\
0,-0.584873,\
-1.871935,0],dtype=np.float128)

helium_1 = np.array([0,0,2,0,-1,0,0,0,0,.95,0,-1])
helium_2 = np.array([0,0,3,0,1,0,0,0,0,.4,0,-1])
p4 = {'m':np.array([1,1,1,1]),'G':1,'dimension':2,'force':1,'fix_first':False}
p3 = {'m':np.array([1,1,1]),'G':1,'dimension':2,'force':1,'fix_first':False}
p_he = {'m':np.array([2,-1,-1]),'G':1,'dimension':2,'force':1,'fix_first':True}
