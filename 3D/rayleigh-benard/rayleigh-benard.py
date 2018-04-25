from __future__ import print_function
from __future__ import division
"""
test: True
"""
from six.moves import range
import numpy as np
import sympy as sp
import mpi4py.MPI as mpi
import pyLBM

X, Y, Z = sp.symbols('X, Y, Z')
rho, qx, qy, qz, T, LA = sp.symbols('rho, qx, qy, qz, T, LA', real=True)

# parameters
dx = 1./128
la = 1
cs = la/np.sqrt(3)

Tu = -0.5
Td =  0.5

Ra = 1e6
Pr = 0.71
g = 9.81
tau = 1./1.8
nu = (2*tau-1)/6*la*dx

diffusivity = nu/Pr
taup = .5+2*diffusivity/la/dx

DeltaT = Td - Tu
xmin, xmax, ymin, ymax, zmin, zmax = 0., 2., 0., 1., 0., 2.
H = ymax - ymin
beta = Ra*nu*diffusivity/(g*DeltaT*H**3)

sf = [0]*4 + [1./tau]*15
sT = [0] + [1./taup]*5

def init_T(x, y, z):
    #ones = np.ones((x.size, y.size, z.size))
    #T = ( Tu*ones*(y>=.1) 
    #    + Td*ones*(y<.1)
    #    + 2*Td*ones*(y>=.1)*np.logical_and(y<.25, ((x-1)**2+(z-1)**2)<.1**2)
    #    )
    #return T
    return Td + (Tu-Td)/(ymax-ymin)*(y-ymin) + (Td-Tu) * (0.1*np.random.random_sample((x.shape[0],y.shape[1],z.shape[2]))-0.5)

def bc_up(f, m, x, y, z):
    m[qx] = 0.
    m[qy] = 0.
    m[qz] = 0.
    m[T] = Tu

def bc_down(f, m, x, y, z):
    m[qx] = 0.
    m[qy] = 0.
    m[qz] = 0.
    m[T] = Td

def save(sol, im):
    x, y, z = sol.domain.x, sol.domain.y, sol.domain.z
    h5 = pyLBM.H5File(sol.mpi_topo, 'rayleigh-benard', './rayleigh-benard', im)
    h5.set_grid(x, y, z)
    h5.add_scalar('T', sol.m[T])
    h5.save()

def feq_NS(v, u):
    cs2 = sp.Rational(1, 3)
    x, y, z = sp.symbols('x, y, z')
    vsymb = sp.Matrix([x, y, z])
    w = sp.Matrix([sp.Rational(1, 3)] + [sp.Rational(1, 18)]*6 + [sp.Rational(1, 36)]*12)
    f = rho + u.dot(vsymb)/cs2 + u.dot(vsymb)**2/(2*cs2**2) - u.norm()**2/(2*cs2)
    return sp.Matrix([w[iv]*f.subs([(x, vv[0]), (y, vv[1]), (z, vv[2])]) for iv, vv in enumerate(v)])

def feq_T(v, u):
    c0 = 1#LA
    x, y, z = sp.symbols('x, y, z')
    vsymb = sp.Matrix([x, y, z])
    f = T/6*(1 + 2*u.dot(vsymb)/c0)
    return sp.Matrix([f.subs([(x, vv[0]), (y, vv[1]), (z, vv[2])]) for iv, vv in enumerate(v)])

def run(dx, Tf, generator="cython", sorder=None, withPlot=True):
    """
    Parameters
    ----------

    dx: double
        spatial step

    Tf: double
        final time

    generator: pyLBM generator

    sorder: list
        storage order

    withPlot: boolean
        if True plot the solution otherwise just compute the solution

    """
    r = X**2+Y**2+Z**2

    dico = {
        'box':{'x':[xmin, xmax], 'y':[ymin, ymax], 'z':[zmin, zmax], 'label':[-1, -1, 0, 1, -1, -1]},
        'space_step':dx,
        'scheme_velocity':la,
        'schemes':[
            {
                'velocities':list(range(19)),
                'conserved_moments': [rho, qx, qy, qz],
                'polynomials':[
                    1,
                    X, Y, Z,
                    19*r - 30,
                    2*X**2 - Y**2 - Z**2,
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
                    (2*X**2 - Y**2 - Z**2)*(3*r - 5),
                    (Y**2 - Z**2)*(3*r - 5),
                    -sp.Rational(53,2)*r + sp.Rational(21,2)*r**2 + 12
                ],
                'relaxation_parameters':sf,
                'feq':(feq_NS, (sp.Matrix([qx, qy, qz]),)),
                'source_terms':{qy: beta*g*T},
                'init':{rho: 1., qx: 0., qy: 0., qz: 0.},
            },
            {
                'velocities':list(range(1,7)),
                'conserved_moments': [T],
                'polynomials':[1, X, Y, Z, 
                               X**2 - Y**2,
                               Y**2 - Z**2,
                               ],
                'feq':(feq_T, (sp.Matrix([qx, qy, qz]),)),
                'relaxation_parameters':sT,
                'init':{T:(init_T,)},
            },
        ],
        'boundary_conditions':{
            0:{'method':{0: pyLBM.bc.Bouzidi_bounce_back, 1: pyLBM.bc.Bouzidi_anti_bounce_back}, 'value':bc_down},
            1:{'method':{0: pyLBM.bc.Bouzidi_bounce_back, 1: pyLBM.bc.Bouzidi_anti_bounce_back}, 'value':bc_up},
        },
        'generator': "cython",
        'parameters': {LA: la},
    }

    sol = pyLBM.Simulation(dico)

    im = 0
    compt = 0
    while sol.t < Tf:
        sol.one_time_step()
        compt += 1
        if compt == 128:
            im += 1
            save(sol, im)
            compt = 0
    return sol

if __name__ == '__main__':
    Tf = 400.
    run(dx, Tf)
