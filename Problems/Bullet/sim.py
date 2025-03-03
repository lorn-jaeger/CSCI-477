import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fsolve
from prettytable import PrettyTable
import math

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


def projectile(t, y, b):
  x, y, vx, vy = y
  speed = np.sqrt(vx ** 2 + vy ** 2)

  g = b.g
  drag = b.get_drag(speed)

  u_x = vx / speed if speed != 0 else 0
  u_y = vy / speed if speed != 0 else 0

  ax = -drag * u_x
  ay = g - drag * u_y

  return np.array([vx, vy, ax, ay])

def plot_reference():
  plt.figure(figsize=(5, 3))
  plt.plot(G1[:, 0], G1[:, 1], 'black', label='G1')
  plt.plot(G7[:, 0], G7[:, 1], 'r-', label='G7')
  plt.axvline(x=1.0, color='k', linestyle='--', label='Speed of Sound')
  plt.ylabel('Cd')
  plt.xlabel('Mach Number')
  plt.title('Reference Projectiles')
  plt.legend()
  plt.grid(True)


class BallisticModel:
  def __init__(self, bc, position, velocity, reference, manufacturer_data, units="imperial"):
    self.units = units
    self.reference = reference
    self.man_x, self.man_v, self.man_y = manufacturer_data

    self.x0, self.y0, = position
    self.vx0, self.vy0 = velocity

    self.bc = self._get_bc(bc)
    self.rho = self._get_rho()
    self.vs = self._get_vs()
    self.g = self._get_g()
    self.cd = self._get_cd()

    self.x, self.y = None, None
    self.vx, self.vy = None, None

  def solve_trajectory(self):
    y0 = np.array([self.x0, self.y0, self.vx0, self.vy0])
    _, state = solve_ode(projectile, (0, 1.7), y0, EulerRichardson, self)
    self.x, self.y, self.vx, self.vy = state.T

  def sight(self, target=300):
    x = target
    y = 0

    def error_function(theta):
      v0 = np.sqrt(self.vx0 ** 2 + self.vy0 ** 2)
      self.vx0 = v0 * np.cos(theta.item())
      self.vy0 = v0 * np.sin(theta.item())

      self.solve_trajectory()

      idx = np.argmin(np.abs(self.x - x))
      return self.y[idx] - y

    initial_guess = np.arctan2(self.vy0, self.vx0)

    optimal = fsolve(error_function, initial_guess)[0]

    v0 = np.sqrt(self.vx0 ** 2 + self.vy0 ** 2)
    self.vx0 = v0 * np.cos(optimal)
    self.vy0 = v0 * np.sin(optimal)

    self.solve_trajectory()

  def optimize(self):
    def total_y_error(theta):
      v0 = np.sqrt(self.vx0 ** 2 + self.vy0 ** 2)

      vx_test = v0 * np.cos(theta)
      vy_test = v0 * np.sin(theta)

      y0 = np.array([self.x0, self.y0, vx_test, vy_test])
      _, state = solve_ode(projectile, (0, 10), y0, EulerRichardson, self)
      x_test, y_test, _, _ = state.T

      total_error = 0

      for i in range(len(self.man_x)):
        if self.units == "imperial":
          x_target = self.man_x[i] * 3
          y_man = self.man_y[i] / 12
        else:
          x_target = self.man_x[i]
          y_man = self.man_y[i]

        if x_target > max(x_test):
          continue

        idx = np.argmin(np.abs(x_test - x_target))
        total_error += (y_test[idx] - y_man) ** 2

      global e
      e = total_error
      return total_error

    result = minimize_scalar(total_y_error, bounds=(0, np.radians(15)), method='bounded')
    optimal_theta = result.x
    v0 = np.sqrt(self.vx0 ** 2 + self.vy0 ** 2)
    self.vx0 = v0 * np.cos(optimal_theta)
    self.vy0 = v0 * np.sin(optimal_theta)

    print(f"Optimal angle: {np.degrees(optimal_theta):.2f} degrees")
    print(f" MSE: {e}")
    self.solve_trajectory()

  def _get_bc(self, bc):
    if self.units == "metric":
      return bc * 0.703  #            lb/in^2 to kg/m^2
    if self.units == "imperial":
      return bc * 144  # lb/in^2 to lb/ft^2

  def _get_rho(self):
    if self.units == "metric":
      return 1.225
    if self.units == "imperial":
      return 0.0742

  def _get_vs(self):
    if self.units == "metric":
      return 343.0
    if self.units == "imperial":
      return 1125.0

  def _get_g(self):
    if self.units == "metric":
      return -9.81
    if self.units == "imperial":
      return -32.17

  def _get_cd(self):
    return interp1d(self.reference[:, 0], self.reference[:, 1], bounds_error=False, fill_value="extrapolate")

  def get_drag(self, speed):
    mach = speed / self.vs
    cd = self.cd(mach)
    drag = 0.5 * (1 / self.bc) * self.rho * speed ** 2 * cd
    return drag

  def plot_velocity(self, title):
    if self.units == "imperial":
      x = self.x / 3
      vel = self.vx
      x_unit, y_unit = "(yards)", "(ft/s)"
    else:
      x = self.x * 1000
      vel = self.vx
      x_unit, y_unit = "(m)", "(m/s)"

    plt.figure()
    plt.plot(x, vel, 'r-', label='Simulated Velocity')
    plt.scatter(self.man_x, self.man_v, color='black', s=10, label='Manufacturer Data')
    plt.xlabel("Distance " + x_unit)
    plt.ylabel("Velocity " + y_unit)
    plt.title(title + " Velocity Fit")

    plt.legend()
    plt.grid()
    plt.show()

  def plot_position(self, title):
    if self.units == "imperial":
      x = self.x / 3
      y = self.y * 12
      y_unit, x_unit = "(in)", "(yards)"
    else:
      x = self.x * 1000
      y = self.y
      y_unit, x_unit = "(m)", "(m)"

    plt.figure()
    plt.plot(x, y, 'b-', label='Simulated Path')
    plt.scatter(self.man_x, self.man_y, color='b', s=10, label='Manufacturer Data')
    plt.xlabel("Distance " + x_unit)
    plt.ylabel("Trajectory " + y_unit)
    plt.legend()
    plt.grid()
    plt.title(title + " Position Fit")
    plt.show()

  def print_errors(self):
    distances = []
    man_y_values = []
    y_errors = []
    man_vx_values = []
    vx_errors = []

    for i in range(len(self.man_x)):
      x_target = self.man_x[i] * 3

      if x_target > max(self.x):
        continue

      idx = np.argmin(np.abs(self.x - x_target))
      y_sim = self.y[idx] * 12
      vx_sim = self.vx[idx]

      y_man = self.man_y[i]
      vx_man = self.man_v[i]

      y_diff = y_sim - y_man
      vx_diff = vx_sim - vx_man

      distances.append(f"{self.man_x[i]:.0f}")
      man_y_values.append(f"{y_man:.3f}")
      y_errors.append(f"{y_diff:.3f}")
      man_vx_values.append(f"{vx_man:.3f}")
      vx_errors.append(f"{vx_diff:.3f}")

    table = PrettyTable()
    table.field_names = ["Distance (yards)"] + distances
    table.add_row(["Drop (in)"] + man_y_values)
    table.add_row(["Drop Error (in)"] + y_errors)
    table.add_row(["Speed (ft/s)"] + man_vx_values)
    table.add_row(["Speed Error (ft/s)"] + vx_errors)

    print(table)


def simple_gravity(t, y, g):
  """
  This describes the ODEs for the kinematic equations:
  dy/dt =  v
  dv/dt = -g
  """
  return np.array([y[1], -g])

def fb_general_drag(t, y, g, v_t, alpha):
  return np.array([y[1], g * (1 - (y[1] / v_t) ** alpha)])


def preprocessing(reference):
  reference[:, 1] = reference[:, 1] * (math.pi / 4)
  return reference

G1 = np.array([[0.00, 0.2629],
          [0.05, 0.2558],
          [0.10, 0.2487],
          [0.15, 0.2413],
          [0.20, 0.2344],
          [0.25, 0.2278],
          [0.30, 0.2214],
          [0.35, 0.2155],
          [0.40, 0.2104],
          [0.45, 0.2061],
          [0.50, 0.2032],
          [0.55, 0.2020],
          [0.60, 0.2034],
          [0.70, 0.2165],
          [0.725, 0.2230],
          [0.75, 0.2313],
          [0.775, 0.2417],
          [0.80, 0.2546],
          [0.825, 0.2706],
          [0.85, 0.2901],
          [0.875, 0.3136],
          [0.90, 0.3415],
          [0.925, 0.3734],
          [0.95, 0.4084],
          [0.975, 0.4448],
          [1.0, 0.4805],
          [1.025, 0.5136],
          [1.05, 0.5427],
          [1.075, 0.5677],
          [1.10, 0.5883],
          [1.125, 0.6053],
          [1.15, 0.6191],
          [1.20, 0.6393],
          [1.25, 0.6518],
          [1.30, 0.6589],
          [1.35, 0.6621],
          [1.40, 0.6625],
          [1.45, 0.6607],
          [1.50, 0.6573],
          [1.55, 0.6528],
          [1.60, 0.6474],
          [1.65, 0.6413],
          [1.70, 0.6347],
          [1.75, 0.6280],
          [1.80, 0.6210],
          [1.85, 0.6141],
          [1.90, 0.6072],
          [1.95, 0.6003],
          [2.00, 0.5934],
          [2.05, 0.5867],
          [2.10, 0.5804],
          [2.15, 0.5743],
          [2.20, 0.5685],
          [2.25, 0.5630],
          [2.30, 0.5577],
          [2.35, 0.5527],
          [2.40, 0.5481],
          [2.45, 0.5438],
          [2.50, 0.5397],
          [2.60, 0.5325],
          [2.70, 0.5264],
          [2.80, 0.5211],
          [2.90, 0.5168],
          [3.00, 0.5133],
          [3.10, 0.5105],
          [3.20, 0.5084],
          [3.30, 0.5067],
          [3.40, 0.5054],
          [3.50, 0.5040],
          [3.60, 0.5030],
          [3.70, 0.5022],
          [3.80, 0.5016],
          [3.90, 0.5010],
          [4.00, 0.5006],
          [4.20, 0.4998],
          [4.40, 0.4995],
          [4.60, 0.4992],
          [4.80, 0.4990],
          [5.00, 0.4988]])

G1 = preprocessing(G1)

G7 = np.array([[0.00, 0.1198],
               [0.05, 0.1197],
               [0.10, 0.1196],
               [0.15, 0.1194],
               [0.20, 0.1193],
               [0.25, 0.1194],
               [0.30, 0.1194],
               [0.35, 0.1194],
               [0.40, 0.1193],
               [0.45, 0.1193],
               [0.50, 0.1194],
               [0.55, 0.1193],
               [0.60, 0.1194],
               [0.65, 0.1197],
               [0.70, 0.1202],
               [0.725, 0.1207],
               [0.75, 0.1215],
               [0.775, 0.1226],
               [0.80, 0.1242],
               [0.825, 0.1266],
               [0.85, 0.1306],
               [0.875, 0.1368],
               [0.90, 0.1464],
               [0.925, 0.1660],
               [0.95, 0.2054],
               [0.975, 0.2993],
               [1.0, 0.3803],
               [1.025, 0.4015],
               [1.05, 0.4043],
               [1.075, 0.4034],
               [1.10, 0.4014],
               [1.125, 0.3987],
               [1.15, 0.3955],
               [1.20, 0.3884],
               [1.25, 0.3810],
               [1.30, 0.3732],
               [1.35, 0.3657],
               [1.40, 0.3580],
               [1.50, 0.3440],
               [1.55, 0.3376],
               [1.60, 0.3315],
               [1.65, 0.3260],
               [1.70, 0.3209],
               [1.75, 0.3160],
               [1.80, 0.3117],
               [1.85, 0.3078],
               [1.90, 0.3042],
               [1.95, 0.3010],
               [2.00, 0.2980],
               [2.05, 0.2951],
               [2.10, 0.2922],
               [2.15, 0.2892],
               [2.20, 0.2864],
               [2.25, 0.2835],
               [2.30, 0.2807],
               [2.35, 0.2779],
               [2.40, 0.2752],
               [2.45, 0.2725],
               [2.50, 0.2697],
               [2.55, 0.2670],
               [2.60, 0.2643],
               [2.65, 0.2615],
               [2.70, 0.2588],
               [2.75, 0.2561],
               [2.80, 0.2533],
               [2.85, 0.2506],
               [2.90, 0.2479],
               [2.95, 0.2451],
               [3.00, 0.2424],
               [3.10, 0.2368],
               [3.20, 0.2313],
               [3.30, 0.2258],
               [3.40, 0.2205],
               [3.50, 0.2154],
               [3.60, 0.2106],
               [3.70, 0.2060],
               [3.80, 0.2017],
               [3.90, 0.1975],
               [4.00, 0.1935],
               [4.20, 0.1861],
               [4.40, 0.1793],
               [4.60, 0.1730],
               [4.80, 0.1672],
               [5.00, 0.1618]])
G7 = preprocessing(G7)


