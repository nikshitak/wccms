# Import some useful modules.
import jax
import jax.numpy as np
import os

# Import JAX-FEM specific modules.
from jax_fem.problem_abc import Problem
from jax_fem.solver_abc import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh #, get_meshio_cell_type, Mesh
from jax_fem.fe_abc import FiniteElement
from jax_fem.iga import BSpline

jax.config.update("jax_debug_nans", True)

# Define constitutive relationship.
class HyperElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first 
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad) 
    # to obtain the 'P_fn' function.
    def get_tensor_map(self):

        def psi(F):
            E = 10.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            #Jinv = J**(-2. / 3.)
            Jinv = 1/(np.cbrt(J**2))
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(u_grad.shape[0])
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress

# Specify mesh-related information (first-order hexahedron element).


domain = 'linear bspline'
Lx, Ly, Lz = 1., 1., 1.
data_dir = os.path.join(os.path.dirname(__file__), 'data')

match domain:
    case 'linear hex':
        ele_type = 'hexahedron'
        mesh = box_mesh(Nx=10,
                       Ny=10,
                       Nz=10,
                       Lx=Lx,
                       Ly=Ly,
                       Lz=Lz,
                       data_dir=data_dir,
                       ele_type=ele_type)
        fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 3)

    case 'quadratic hex':
        ele_type = 'hexahedron27'
        mesh = box_mesh(Nx=6,
                       Ny=6,
                       Nz=6,
                       Lx=Lx,
                       Ly=Ly,
                       Lz=Lz,
                       data_dir=data_dir,
                       ele_type=ele_type)
        fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 6)

    case 'cubic hex':
        ele_type = 'hexahedron64'
        mesh = box_mesh(Nx=6,
                       Ny=6,
                       Nz=6,
                       Lx=Lx,
                       Ly=Ly,
                       Lz=Lz,
                       data_dir=data_dir,
                       ele_type=ele_type)
        fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 9)

    case 'quartic hex':
        ele_type = 'hexahedron125'
        mesh = box_mesh(Nx=2,
                       Ny=2,
                       Nz=2,
                       Lx=Lx,
                       Ly=Ly,
                       Lz=Lz,
                       data_dir=data_dir,
                       ele_type=ele_type)
        fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 3*4)
    case 'quintic hex':
        ele_type = 'hexahedron216'
        mesh = box_mesh(Nx=2,
                       Ny=2,
                       Nz=2,
                       Lx=Lx,
                       Ly=Ly,
                       Lz=Lz,
                       data_dir=data_dir,
                       ele_type=ele_type)
        fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 3*5)
    case 'linear bspline':
        deg = 1
        ele_type = 'SPLINEHEX'+str(deg)
        knot0 = np.hstack((np.zeros(deg), np.linspace(0,1,11) ,np.ones(deg)))
        knot1 = np.hstack((np.zeros(deg), np.linspace(0,1,11) ,np.ones(deg)))
        knot2 = np.hstack((np.zeros(deg), np.linspace(0,1,11) ,np.ones(deg)))
        knots = [knot0, knot1, knot2]
        degrees = 3*[deg]
        fe = BSpline(knots, degrees, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 3*deg)
    case 'quadratic bspline':
        deg = 2
        ele_type = 'SPLINEHEX'+str(deg)
        knot0 = np.hstack((np.zeros(deg), np.linspace(0,1,5) ,np.ones(deg)))
        knot1 = np.hstack((np.zeros(deg), np.linspace(0,1,5) ,np.ones(deg)))
        knot2 = np.hstack((np.zeros(deg), np.linspace(0,1,5) ,np.ones(deg)))
        knots = [knot0, knot1, knot2]
        degrees = 3*[deg]
        fe = BSpline(knots, degrees, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 3*deg)
    case 'cubic bspline':
        deg = 3
        ele_type = 'SPLINEHEX'+str(deg)
        knot0 = np.hstack((np.zeros(deg), np.linspace(0,1,2) ,np.ones(deg)))
        knot1 = np.hstack((np.zeros(deg), np.linspace(0,1,2) ,np.ones(deg)))
        knot2 = np.hstack((np.zeros(deg), np.linspace(0,1,2) ,np.ones(deg)))
        knots = [knot0, knot1, knot2]
        degrees = 3*[deg]
        fe = BSpline(knots, degrees, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 3*deg)
    case 'genhex':
        deg = 3
        ele_type = 'GENHEX'+str(deg)
        cell_type = get_meshio_cell_type(ele_type)
        meshio_mesh = box_mesh(Nx=2,
                       Ny=2,
                       Nz=2,
                       Lx=Lx,
                       Ly=Ly,
                       Lz=Lz,
                       data_dir=data_dir,
                       ele_type=ele_type)
        mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
        fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 3*deg)
    case 'splinehex':
        deg = 2
        ele_type = 'SPLINEHEX'+str(deg)
        knot0 = np.hstack((np.zeros(deg), np.linspace(0,1,5) ,np.ones(deg)))
        knot1 = np.hstack((np.zeros(deg), np.linspace(0,1,5) ,np.ones(deg)))
        knot2 = np.hstack((np.zeros(deg), np.linspace(0,1,5) ,np.ones(deg)))
        knots = [knot0, knot1, knot2]
        degrees = 3*[deg]
        bspl = BSpline(knots, degrees, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 3*deg)
        mesh = bspl.mesh
        fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 3*deg)


#mesh1 = Mesh(meshio_mesh1.points, meshio_mesh1.cells_dict[cell_type1])
#fe1 = FiniteElement(mesh1, vec = 3, dim = 3, ele_type = ele_type1, gauss_order = 3)

# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)


def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)


# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.


def dirichlet_val_x2(point):
    return (0.5 + (point[1] - 0.5) * np.cos(np.pi / 3.) -
            (point[2] - 0.5) * np.sin(np.pi / 3.) - point[1]) / 2.


def dirichlet_val_x3(point):
    return (0.5 + (point[1] - 0.5) * np.sin(np.pi / 3.) +
            (point[2] - 0.5) * np.cos(np.pi / 3.) - point[2]) / 2.


dirichlet_bc_info = [[left] * 3 + [right] * 3, [0, 1, 2] * 2,
                     [zero_dirichlet_val, dirichlet_val_x2, dirichlet_val_x3] +
                     [zero_dirichlet_val] * 3]


problem = HyperElasticity(fe,
                          #fe_bindings = [[0,1]],
                          dirichlet_bc_info=dirichlet_bc_info)


# Solve the defined problem.
sol = solver(problem, use_petsc=False)
# Store the solution to local file.
#vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
#save_sol(problem.fes[0], sol.reshape(-1, 3), vtk_path)