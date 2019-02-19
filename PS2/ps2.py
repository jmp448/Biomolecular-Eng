import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parameters
rxt = 19.8  #uM
rlt = 123.96  #uM

mu = 0.000385  #s^-1

g = 7.07e-13  #uM

kappa_x = 0.0048  #uM
kappa_l = kappa_x

tau_x1 = 0.875  #s^-1
tau_x2 = 0.4375  #s^-1
tau_x3 = 1.75  #s^-1

tau_l1 = tau_x1
tau_l2 = tau_x2
tau_l3 = tau_x3

k_xel1 = 0.035  #s^-1
k_xel2 = 0.0175  #s^-1
k_xel3 = 0.07  #s^-1

k_xdeg1 = 5.5e-3  #s^-1
k_xdeg2 = 5.5e-3  #s^-1
k_xdeg3 = 5.5e-3  #s^-1

k_xdegex = k_xdeg1 + mu  #s^-1

k_lel1 = 0.03625  #s^-1
k_lel2 = 0.0181  #s^-1
k_lel3 = 0.0725  #s^-1

k_ldeg1 = 3.85e-6  #s^-1
k_ldeg2 = 3.85e-6  #s^-1
k_ldeg3 = 3.85e-6  #s^-1

k_ldegex = k_ldeg1 + mu  #s^-1

w_basal = 0.00001
w_i = 300

n1 = 1.5
n2 = 1.5
n3 = 1.5


def eqn(y, t, ind, broken):
    m1, m2, m3, p1, p2, p3 = y

    A = [
    [-k_xdegex, 0, 0, 0, 0, 0],
    [0, -k_xdeg2 - mu, 0, 0, 0, 0],
    [0, 0, -k_xdeg3 - mu, 0, 0, 0],
    [0, 0, 0, -k_ldegex, 0, 0],
    [0, 0, 0, 0, -k_ldeg2 - mu, 0],
    [0, 0, 0, 0, 0, -k_ldeg3 - mu]
    ]

    S = [
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
    ]

    rx1 = k_xel1 * rxt * g / (tau_x1*kappa_x + g*(tau_x1+1))
    rx2 = k_xel2 * rxt * g / (tau_x2*kappa_x + g*(tau_x2+1))
    rx3 = k_xel3 * rxt * g / (tau_x3*kappa_x + g*(tau_x3+1))

    rl1 = k_lel1 * rlt * m1 / (tau_l1*kappa_l + m1*(tau_l1+1))
    rl2 = k_lel2 * rlt * m2 / (tau_l2*kappa_l + m2*(tau_l2+1))
    rl3 = k_lel3 * rlt * m3 / (tau_l3*kappa_l + m3*(tau_l3+1))

    i_conc = ind  #uM
    f_i = i_conc**n1 / (kappa_x**n1 + i_conc**n1)

    w_12 = 1
    w_32 = 1.5
    w_13 = 2
    if broken:
        w_23 = 0
    else:
        w_23 = 3

    f_12 = p1**n2 / (kappa_x**n2 + p1**n2)
    f_32 = p3**n2 / (kappa_x**n2 + p3**n2)

    f_13 = p1**n3 / (kappa_x**n3 + p1**n3)
    f_23 = p2**n3 / (kappa_x**n3 + p2**n3)

    ux1 = (w_basal + w_i * f_i)/(1 + w_basal + w_i * f_i)
    ux2 = (w_basal + w_12 * f_12 + w_32 * f_32)/ (1 + w_basal + w_12 * f_12 + w_32 * f_32)
    ux3 = (w_basal + w_13 * f_13 + w_23 * f_23)/ (1 + w_basal + w_13 * f_13 + w_23 * f_23)

    r = [rx1*ux1, rx2*ux2, rx3*ux3, rl1, rl2, rl3]

    dxdt = np.matmul(A, y) + np.matmul(S, r)

    return dxdt


def solve_ode(broken=False):
    t_1 = np.linspace(0, 3600, 1000)
    y_on = [0, 0, 0, 0, 0, 0]
    if broken:
        sol1 = odeint(eqn, y_on, t_1, args=(10,True))
    else:
        sol1 = odeint(eqn, y_on, t_1, args=(10,False))

    t_2 = np.linspace(3600, 360*60, 6000)
    y_off = sol1[len(sol1)-1,:]
    if broken:
        sol2 = odeint(eqn, y_off, t_2, args=(0,True))
    else:
        sol2 = odeint(eqn, y_off, t_2, args=(0,False))

    return sol1, sol2


def plot_mRNA(sol1, sol2):
    t_1 = np.linspace(0, 3600, 1000)
    t_2 = np.linspace(3600, 360*60, 6000)
    t = np.concatenate((t_1, t_2), 0)

    m1 = np.concatenate((sol1[:,0], sol2[:,0]))
    m2 = np.concatenate((sol1[:,1], sol2[:,1]))
    m3 = np.concatenate((sol1[:,2], sol2[:,2]))

    ind_on = np.full(1000, m1[1000]/2)
    ind_off = np.zeros(6000)
    ind = np.concatenate((ind_on, ind_off), 0)

    plt.plot(t, m1, label="m1")
    plt.plot(t, m2, label="m2")
    plt.plot(t, m3, label="m3")
    plt.plot(t, ind, '--r', label="inducer")
    plt.legend(loc="best")
    plt.show()


def plot_proteins(sol1, sol2):
    t_1 = np.linspace(0, 3600, 1000)
    t_2 = np.linspace(3600, 360*60, 6000)
    t = np.concatenate((t_1, t_2), 0)

    p1 = np.concatenate((sol1[:,3], sol2[:,3]))
    p2 = np.concatenate((sol1[:,4], sol2[:,4]))
    p3 = np.concatenate((sol1[:,5], sol2[:,5]))

    ind_on = np.full(1000, p1[1000]/2)
    ind_off = np.zeros(6000)
    ind = np.concatenate((ind_on, ind_off), 0)

    plt.plot(t, p1, label="p1")
    plt.plot(t, p2, label="p2")
    plt.plot(t, p3, label="p3")
    plt.plot(t, ind, 'r--', label="inducer")
    plt.legend(loc="best")
    plt.show()


def solve_discretized(ind, tau=0.6, broken=False):
    A = np.array([
    [-k_xdegex, 0, 0, 0, 0, 0],
    [0, -k_xdeg2 - mu, 0, 0, 0, 0],
    [0, 0, -k_xdeg3 - mu, 0, 0, 0],
    [0, 0, 0, -k_ldegex, 0, 0],
    [0, 0, 0, 0, -k_ldeg2 - mu, 0],
    [0, 0, 0, 0, 0, -k_ldeg3 - mu]
    ])

    A_hat = np.exp(tau*A)

    A_inv = np.linalg.inv(A)

    S = [
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
    ]

    I = [
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
    ]

    i_conc = ind  #uM
    f_i = i_conc**n1 / (kappa_x**n1 + i_conc**n1)

    w_12 = 1200
    w_32 = 15000
    w_13 = 1300
    if broken:
        w_23 = 0
    else:
        w_23 = 10000

    S_hat = np.matmul(A_inv, np.matmul(A_hat-I, S))

    # r initial
    rx1 = k_xel1 * rxt * g / (tau_x1*kappa_x + g*(tau_x1+1))
    rx2 = k_xel2 * rxt * g / (tau_x2*kappa_x + g*(tau_x2+1))
    rx3 = k_xel3 * rxt * g / (tau_x3*kappa_x + g*(tau_x3+1))

    ux1 = (w_basal + w_i * f_i)/(1 + w_basal + w_i * f_i)
    ux2 = (w_basal)/ (1 + w_basal)
    ux3 = (w_basal)/ (1 + w_basal)

    r = [rx1*ux1, rx2*ux2, rx3*ux3, 0, 0, 0]

    x = np.zeros([6000, 6])
    for i in range(1, 6):
        x[i,:] = np.matmul(A_hat, x[i-1,:])+np.matmul(S_hat, r)
        [m1, m2, m3, p1, p2, p3] = x[i,:]
        print(p1)

        rx1 = k_xel1 * rxt * g / (tau_x1*kappa_x + g*(tau_x1+1))
        rx2 = k_xel2 * rxt * g / (tau_x2*kappa_x + g*(tau_x2+1))
        rx3 = k_xel3 * rxt * g / (tau_x3*kappa_x + g*(tau_x3+1))

        rl1 = k_lel1 * rlt * m1 / (tau_l1*kappa_l + m1*(tau_l1+1))
        rl2 = k_lel2 * rlt * m2 / (tau_l2*kappa_l + m2*(tau_l2+1))
        rl3 = k_lel3 * rlt * m3 / (tau_l3*kappa_l + m3*(tau_l3+1))

        f_12 = p1**n2 / (kappa_x**n2 + p1**n2)
        print("Denominator: %.4f + %.4f = %.4f" % (kappa_x**n2, p1, kappa_x**n2 + p1**n2))
        f_32 = p3**n2 / (kappa_x**n2 + p3**n2)

        f_13 = p1**n3 / (kappa_x**n3 + p1**n3)
        f_23 = p2**n3 / (kappa_x**n3 + p2**n3)

        ux1 = (w_basal + w_i * f_i)/(1 + w_basal + w_i * f_i)
        ux2 = (w_basal + w_12 * f_12 + w_32 * f_32)/ (1 + w_basal + w_12 * f_12 + w_32 * f_32)
        ux3 = (w_basal + w_13 * f_13 + w_23 * f_23)/ (1 + w_basal + w_13 * f_13 + w_23 * f_23)

        r = [rx1*ux1, rx2*ux2, rx3*ux3, rl1, rl2, rl3]

    t = np.linspace(0, 3600, 6000)
    m1 = x[:,0]
    m2 = x[:,1]
    m3 = x[:,2]
    p1 = x[:,3]
    p2 = x[:,4]
    p3 = x[:,5]

    plt.plot(t, m1, label="m1")
    plt.plot(t, m2, label="m2")
    plt.plot(t, m3, label="m3")
    plt.legend(loc="best")
    plt.show()

    plt.plot(t, p1, label="p1")
    plt.plot(t, p2, label="p2")
    plt.plot(t, p3, label="p3")
    plt.legend(loc="best")
    plt.show()


def main():
    sol1, sol2 = solve_ode(broken=True)
    plot_mRNA(sol1, sol2)
    plot_proteins(sol1, sol2)

    # solve_discretized(10, tau=0.6, broken=False)



if __name__ == "__main__":
    main()
