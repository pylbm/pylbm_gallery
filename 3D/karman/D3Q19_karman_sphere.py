
import numpy as np
import sympy as sp
import pylbm
import sys

X, Y, Z, LA = sp.symbols('X, Y, Z, LA')
rho, qx, qy, qz = sp.symbols('rho, qx, qy, qz', real=True)

def feq(v, u):
    cs2 = sp.Rational(1, 3)
    x, y, z = sp.symbols('x, y, z')
    vsymb = sp.Matrix([x, y, z])
    w = sp.Matrix([sp.Rational(1,3)] + [sp.Rational(1, 18)]*6 + [sp.Rational(1, 36)]*12)
    f = rho + u.dot(vsymb)/cs2 + u.dot(vsymb)**2/(2*cs2**2) - u.norm()**2/(2*cs2)
    return sp.Matrix([w[iv]*f.subs([(x, vv[0]), (y, vv[1]), (z, vv[2])]) for iv, vv in enumerate(v)])

def bc_rect(f, m, x, y, z, rhoo, uo):
    m[rho] = 0.
    m[qx] = rhoo*uo
    m[qy] = 0.
    m[qz] = 0.

def plot_vorticity(sol, bornes = False):
    #### vorticity
    ux = sol.m[qx][:,:,3]
    uy = sol.m[qy][:,:,3]
    vort = np.abs(ux[1:-1, 2:] - ux[1:-1, :-2]
                  - uy[2:, 1:-1] + uy[:-2, 1:-1])
    if bornes:
        return vort.T, 0.0, 0.1, 1
    else:
        return vort.T

def save(sol, im):
    x, y, z = sol.domain.x, sol.domain.y, sol.domain.z
    h5 = pylbm.H5File(sol.mpi_topo, 'karman', './karman', im)
    h5.set_grid(x, y, z)
    h5.add_scalar('rho', sol.m[rho])
    qx_n, qy_n, qz_n = sol.m[qx], sol.m[qy], sol.m[qz]
    h5.add_vector('velocity', [qx_n, qy_n, qz_n])
    h5.save()

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1,  barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = '{0:.' + str(decimals) + 'f}'
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '-' * filledLength + ' ' * (barLength - filledLength)
    print('\r{0:s} |{1:s}| {2:s}% {3:s}'.format(prefix, bar, percents,  suffix), end='', file=sys.stdout, flush=True)
    if iteration == total:
        print('', end = '\n', file=sys.stdout, flush=True)

def run(dx, Tf, generator="cython", sorder=None, withPlot=True):
    """
    Parameters
    ----------

    dx: double
        spatial step

    Tf: double
        final time

    generator: pylbm generator

    sorder: list
        storage order

    withPlot: boolean
        if True plot the solution otherwise just compute the solution

    """
    la = 1
    rhoo = 1.
    uo = 0.1
    radius = 0.125

    Re = 2000
    nu = rhoo*uo*2*radius/Re

    #tau = .5*(6*nu/la/dx + 1)
    #print(1./tau)

    s1 = 1.19
    s2 = s10 = 1.4
    s4 = 1.2
    dummy = 3.0/(la*rhoo*dx)
    s9 = 1./(nu*dummy +.5)
    s13 = 1./(nu*dummy +.5)
    s16 = 1.98
    #[0, s1, s2, 0, s4, 0, s4, 0, s4, s9, s10, s9, s10, s13, s13, s13, s16, s16, s16]
    s = [0]*4 + [s1, s9, s9, s13, s13, s13, s4, s4, s4, s16, s16, s16, s10, s10, s2]
    r = X**2+Y**2+Z**2

    d_p = {
        'geometry': {
            'xmin': 0,
            'xmax': 2,
            'ymin': 0,
            'ymax': 1,
            'zmin': 0,
            'zmax': 1
        }
    }

    dico = {
        'box': {
            'x': [0., 2.],
            'y': [0., 1.],
            'z': [0., 1.],
            'label': [0, 1, 0, 0, 0, 0]
        },
        'elements':[pylbm.Sphere((.3, .5+2*dx, .5+2*dx), radius, 2)],
        'space_step': dx,
        'scheme_velocity': la,
        'schemes': [
            {
                'velocities': list(range(19)),
                'conserved_moments': [rho, qx, qy, qz],
                'polynomials': [
                    1,
                    X, Y, Z,
                    19*r - 30,
                    3*X**2 - r,
                    Y**2-Z**2,
                    X*Y, 
                    Y*Z, 
                    Z*X,
                    X*(5*r - 9),
                    Y*(5*r - 9),
                    Z*(5*r - 9),
                    X*(Y**2 - Z**2),
                    Y*(Z**2 - X**2),
                    Z*(X**2 - Y**2),
                    (-2*X**2 + Y**2 + Z**2)*(3*r - 5),
                    -5*Y**2 + 5*Z**2 + 3*X**2*(Y**2 - Z**2) + 3*Y**4 - 3*Z**4,
                    -53*r + 21*r**2 + 24
                ],
                'relaxation_parameters': s,#[0]*4 + [1./tau]*15,
                'feq': (feq, (sp.Matrix([qx, qy, qz]),)),
                'init': {
                    rho: rhoo,
                    qx: rhoo*uo,
                    qy: 0.,
                    qz: 0.
                },
        }],
        'boundary_conditions': {
            0: {'method': {0: pylbm.bc.BouzidiBounceBack}, 'value': (bc_rect, (rhoo, uo))},
            1: {'method': {0: pylbm.bc.NeumannX}},
            2: {'method': {0: pylbm.bc.BouzidiBounceBack}},
        },
        'parameters': {LA: la},
        'generator': generator,
    }

    sol = pylbm.Simulation(dico, sorder=sorder)
    return
    dt = 1./4

    if withPlot:
        #### choice of the plotted field
        plot_field = plot_vorticity

        #### init viewer
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        ax = fig[0]
        ax.xaxis_set_visible(False)
        ax.yaxis_set_visible(False)
        field, ymin, ymax, decalh = plot_field(sol, bornes = True)
        image = ax.image(field, clim=[ymin, ymax], cmap="jet")

        def update(iframe):
            while sol.t < iframe * dt:
                sol.one_time_step()
            image.set_data(plot_field(sol))
            ax.title = "Solution t={0:f}".format(sol.t)


        #### run the simulation
        fig.animate(update, interval=1)
        fig.show()
    else:
        im = 0
        save(sol, im)
        while sol.t < Tf:
            im += 1
            while sol.t < im * dt:
                sol.one_time_step()
            #printProgress(im, int(Tf/dt), prefix = 'Progress:', suffix = 'Complete', barLength =  50)
            save(sol, im)

    return sol

if __name__ == '__main__':
    dx = 1./256
    Tf = 100.
    run(dx, Tf, withPlot=False)
