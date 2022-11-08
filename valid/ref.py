import numpy as np
from scipy.integrate import solve_ivp
from tank_model import ds_dt


def simulate(tf, t_step, h_zero, area, beta, reference, K, u_limits):
  iterations = int(tf / t_step)

  tank_level = np.zeros(iterations)
  control_signal = np.zeros(iterations)
  references = np.zeros(iterations)
  time = np.linspace(.0, tf, iterations)
  errors = np.zeros(iterations)

  h = h_zero

  Kp, Ki, Kd = K[0], K[1], K[2]

  err = reference - h

  accumulator = (beta * np.sqrt(reference)) / Ki

  tank_level[0] = h

  for i in range(iterations):
    t = time[i]
    references[i] = reference

    # Controller

    last_err = err
    err = reference - h

    # control signal
    accumulator += .5 * (err + last_err) * t_step

    u = (Kp * err) + (Ki * accumulator) + (Kd * ((err - last_err) / t_step))
    u = np.clip(u, u_limits[0], u_limits[1])

    control_signal[i] = u

    # END Controller

    # simulate tank
    sol = solve_ivp(
      ds_dt,
      t_span=(t, t + t_step),
      y0=[h],
      t_eval=(t, t + t_step),
      args=(area, beta, u),
      method='RK23'
    )

    tank_level[i] = h + np.random.uniform(-.01, .01)

    h = sol.y[0][-1]

    errors[i] = np.abs(err)

  return tank_level, control_signal, references, np.sqrt(np.mean(errors)), time
