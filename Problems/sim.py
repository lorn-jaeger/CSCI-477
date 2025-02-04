import numpy as np

def Euler(dt, f, t, y, args):
    return y + f(t,y,*args) * dt

def EulerCromer(dt, f, t, y, args):
    y_end = y + f(t, y, *args) * dt
    return y + f(t + dt, y_end, *args) * dt

def EulerRichardson(dt, f, t, y, args):
    k1 = f(t, y, *args)
    y_mid = y + k1 * (dt/2)
    k2 = f(t + dt/2, y_mid, *args)
    return y + k2 * dt

def solve_ode(f,tspan, y0, method = Euler, *args, **options):
    t0, tf = tspan
    dt = options.get('first_step', 0.01)

    t = [t0]
    y = [y0]

    while t[-1] < tf:
        y_c = y[-1]
        t_c = t[-1]

        y_new = method(dt, f, t_c, y_c, args)
        t_new = t_c + dt

        y.append(y_new)
        t.append(t_new)

    return np.array(t), np.array(y)


def simple_gravity(t, y, g):
  """
  This describes the ODEs for the kinematic equations:
  dy/dt =  v
  dv/dt = -g
  """
  return np.array([y[1], -g])

def fb_general_drag(t, y, g, v_t, alpha):
  return np.array([y[1], g * (1 - (y[1] / v_t) ** alpha)])
