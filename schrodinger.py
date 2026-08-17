import sympy as sp
import torch
from physicsnemo.models.fno import FNO
from physicsnemo.sym import PDE, PhysicsInformer

# 1. Define the Physics using SymPy (Nonlinear Schrödinger)
class Schrodinger(PDE):
    def __init__(self, V, kappa=1.0):
        super().__init__()
        # Define symbols for Real (u) and Imaginary (v) wave components
        u, v = sp.Symbol('u'), sp.Symbol('v')
        t, x, y = sp.Symbol('t'), sp.Symbol('x'), sp.Symbol('y')
        
        # Framework handles derivatives automatically
        u_t, v_t = sp.Derivative(u, t), sp.Derivative(v, t)
        lap_u = sp.Derivative(u, x, 2) + sp.Derivative(u, y, 2)
        lap_v = sp.Derivative(v, x, 2) + sp.Derivative(v, y, 2)
        norm_sq = u**2 + v**2
        
        # Add PDE Residuals: i(u_t + i v_t) = [-0.5 * Laplacian + V + kappa|psi|^2](u + iv)
        self.add_equation("real_res", v_t - 0.5 * lap_u + (V + kappa * norm_sq) * u)
        self.add_equation("imag_res", u_t + 0.5 * lap_v - (V + kappa * norm_sq) * v)



# 2. Initialize FNO and Informer
model = FNO(in_channels=3, out_channels=2).cuda() # Inputs: x, y, t. Outputs: u, v.

schrodinger_pde = Schrodinger(V=0.0)

informer = PhysicsInformer(pde=schrodinger_pde, model=model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)