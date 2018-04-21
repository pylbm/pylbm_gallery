from __future__ import print_function, division
import sys
from six.moves import range
import numpy as np
import sympy as sp
import pyLBM
import sys

"""

shallow water system simulated by vectorial solver D2Q[4,4,4]

"""
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
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
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '*' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

VTK_save = True

X, Y, LA = sp.symbols('X, Y, LA')
rho, qx, qy = sp.symbols('rho, qx, qy')

def initialization_rho(x,y):
    return rhoo * np.ones((x.size, y.size)) + deltarho * ((x-0.5*(xmin+xmax))**2+(y-0.5*(ymin+ymax))**2 < 0.25**2)

def save(x, y, m, num):
    if num > 0:
        vtk = pyLBM.VTKFile(filename, path, num)
    else:
        vtk = pyLBM.VTKFile(filename, path, num, init_pvd = True)
    vtk.set_grid(x, y)
    vtk.add_scalar('rho', m[rho])
    vtk.save()

# parameters
rhoo = 1.
deltarho = 1.
Taille = 2.
xmin, xmax, ymin, ymax = -0.5*Taille, 0.5*Taille, -0.5*Taille, 0.5*Taille
la = 4 # velocity of the scheme
g = 1.
sigma = 1.e-4
s_0qx = 2.#1./(0.5+sigma)
s_0xy = 1.5
s_1qx = 1.5
s_1xy = 1.2
s0  = [0., s_0qx, s_0qx, s_0xy]
s1  = [0., s_1qx, s_1qx, s_1xy]
if VTK_save:
    dx = 1./512 # spatial step
else:
    dx = 1./128

vitesse = list(range(1,5))
polynomes = [1, LA*X, LA*Y, X**2-Y**2]

dico   = {
    'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'label':-1},
    'space_step':dx,
    'scheme_velocity':la,
    'parameters':{LA:la},
    'schemes':[
        {
            'velocities':vitesse,
            'conserved_moments':rho,
            'polynomials':polynomes,
            'relaxation_parameters':s0,
            'equilibrium':[rho, qx, qy, 0.],
            'init':{rho:(initialization_rho,)},
        },
        {
            'velocities':vitesse,
            'conserved_moments':qx,
            'polynomials':polynomes,
            'relaxation_parameters':s1,
            'equilibrium':[qx, qx**2/rho + 0.5*g*rho**2, qx*qy/rho, 0.],
            'init':{qx:0.},
        },
        {
            'velocities':vitesse,
            'conserved_moments':qy,
            'polynomials':polynomes,
            'relaxation_parameters':s1,
            'equilibrium':[qy, qy*qx/rho, qy**2/rho + 0.5*g*rho**2, 0.],
            'init':{qy:0.},
        },
    ],
    'generator': pyLBM.generator.CythonGenerator,
}

sol = pyLBM.Simulation(dico)

x, y = sol.domain.x, sol.domain.y

if VTK_save:
    Tf = 20.
    im = 0
    l = Tf / sol.dt / 32
    printProgress(im, l, prefix = 'Progress:', suffix = 'Complete', barLength = 50)
    filename = 'shallow_water'
    path = './data_' + filename
    save(x, y, sol.m, im)
    while sol.t<Tf:
        for k in range(32):
            sol.one_time_step()
        im += 1
        printProgress(im, l, prefix = 'Progress:', suffix = 'Complete', barLength = 50)
        save(x, y, sol.m, im)
else:
    viewer = pyLBM.viewer.matplotlibViewer
    fig = viewer.Fig()
    ax = fig[0]
    im = ax.image(sol.m[rho].transpose(), clim=[rhoo-.5*deltarho, rhoo+1.5*deltarho])
    ax.title = 'solution at t = {0:f}'.format(sol.t)

    def update(iframe):
        for k in range(32):
            sol.one_time_step()      # increment the solution of one time step
        im.set_data(sol.m[rho].transpose())
        ax.title = 'solution at t = {0:f}'.format(sol.t)

    # run the simulation
    fig.animate(update, interval=1)
    fig.show()
