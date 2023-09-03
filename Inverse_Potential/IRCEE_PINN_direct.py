import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
# plotly.offline.init_notebook_mode()
import numpy as np
import matplotlib.pyplot as plt
# from myPDE import IRCEE
import torch
from torch import nn
from scipy.sparse import spdiags
from scipy.fftpack import fft
from scipy.fftpack import ifft
np.random.seed(0)
device = "cuda:0"
"""
geom
"""
class Space1d():
    def __init__(self,a,b):
        self.a=a
        self.b=b

    # def on_boundary(self,xt):
    #     return np.any(np.isclose(xt[:, :-1], [self.a, self.b]), axis=-1)
    # def on_initial(self,xt):
    #     return np.isclose(xt[:, -1:],self.t0).flatten()
    def uniform_points(self,n,boundary=True):
        if boundary:
            x=np.linspace(self.a, self.b, num=n)[:, None]
        else:
            x=np.linspace(self.a, self.b, num=n+1,endpoint=False)[1:, None]
        return x
    def random_points(self,nx):
        xr=np.random.rand(nx)[:, None]
        x=(self.b-self.a)*xr+self.a
        return x
    def uniform_boundary_points(self):
        return np.vstack((self.a, self.b))

N = 100
h = 1/N
domain=Space1d(0,1)
N=100
n=N+1
x=domain.uniform_points(n)
x_torch=torch.from_numpy(x).to(torch.float32).to(device)
"""
Network
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_Tanh_stack_u = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
        self.linear_Tanh_stack_q = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_Tanh_stack_u(x)*x*(1-x),self.linear_Tanh_stack_q(x)
        return logits
  
model = NeuralNetwork().to(device)
loss_fn = nn.MSELoss()

"""
true data
"""
def k(x):
    return 0.2 + 0.8*np.float32((x > 0.25)*(x < 0.75))

ktrue = k(x)
nu = 0.005
# f= -np.sin(2 * np.pi * x)
f = np.sin(2 * np.pi * x)
e = np.ones(N+1)*nu
diag_data = np.array([2*e+ktrue[:, 0]*h**2, -e, -e])
A = spdiags(diag_data, [0, 1, -1], N+1, N+1)
A = A.tocsr()
A[0, 0] = 1
A[0, 1] = 0
A[1, 0] = 0
A[-1, -1] = 1
A[-2, -1] = 0
A[-1, -2] = 0
A = A.todense()
exact_u = np.linalg.solve(A, f*h**2)
"""
ud
"""
noise_sigma = 0.05*np.linalg.norm(exact_u)*h
noise = np.random.randn(N+1, 1)*noise_sigma
ud = exact_u+noise
# plt.figure()
# plt.plot(x, exact_u, "-", label="u_true")
# plt.plot(x, ud, "*-", label="u_noised")
# plt.legend()
# plt.show()
def f_torch(x):
    # k_torch=0.5+0.5*torch.where((x>0.25*torch.ones_like(x))&(x<0.75*torch.ones_like(x)),torch.ones_like(x),torch.zeros_like(x))
    # return nu*4*np.pi**2*torch.sin(2 * np.pi * x)+k_torch*torch.sin(2 * np.pi * x)
    return torch.sin(2 * np.pi * x)
def BC_torch(x):
    return 0*x
"""
trainning function
"""
Quiet=True
def train(model, loss_fn, optimizer):
    optimizer.zero_grad()
    loss = data.losses(model, loss_fn)
    loss.backward()
    optimizer.step()
    # loss= loss.item()
    if not Quiet:
        print(f"loss: {loss:>7f} ") 
def train_BFGS( model, loss_fn, optimizer):
    def closure():
        optimizer.zero_grad()
        loss = data.losses(model, loss_fn)
        loss.backward()
        return loss
    optimizer.step(closure)
    loss= closure().item()
    if not Quiet:
        print(f"loss: {loss:>7f} ") 
    
    """
denoiser
"""
def TVdenoiser(b,alpha):
    diff_k=lambda z:np.vstack((z[1:,:]-z[0:-1,:],z[0,:]-z[-1,:]))
    diffT_k=lambda z:np.vstack((z[-1,:]-z[0,:],z[0:-1,:]-z[1:,:]))
    num=b.size
    dh=np.zeros([num])
    dh[0]=-1
    dh[-1]=1
    dh=fft(dh)
    beta_innner=0.5
    co_D=beta_innner*pow(abs(dh),2)+1
    y=b.copy()
    z=y.copy()
    dual=0*y.copy()
    for i_iter in range(50):
        Temp=diffT_k(beta_innner*z-dual)+b
        y_temp=ifft(fft(Temp[:,0])/co_D)
        y[:,0]=np.float32(y_temp)
        dy=diff_k(y)
        sk=dy+dual/beta_innner
        # nsk=abs(sk)
        # nsk_temp=1-(alpha/nsk)/beta
        z1=np.where(sk<alpha/beta_innner,sk,alpha/beta_innner)
        z=sk-np.where(z1>-alpha/beta_innner,z1,-alpha/beta_innner)
        dual=dual+beta_innner*(dy-z)
    return y

def pde(x_torch,u_torch,k_torch,k_prox):
    x_np=x_torch.cpu().detach().numpy()
    x_np=np.round(x*(n-1))
    i_a=np.int32(x_np)
    k_prox_p=k_prox[i_a,0]
    k_prox_torch=torch.from_numpy(k_prox_p).to(torch.float32).to(device)
    
    du=torch.autograd.grad(u_torch, x_torch, grad_outputs=torch.ones_like(u_torch), create_graph=True)[0]
    ddu=torch.autograd.grad(du, x_torch, grad_outputs=torch.ones_like(du), create_graph=True)[0]
    error_pde=-nu*ddu+k_torch*u_torch-f_torch(x_torch)
    error_prox=np.sqrt(beta)*(k_torch-k_prox_torch)
    return torch.vstack((error_pde,error_prox))
class IRCEE_V2():
    def __init__(
        self,
        geometry,
        pde,
        bcs,
        k_prox,
        num_domain,
        para_bd,
        para_ob,
        observe_x,
        observe_u,
    ):
        self.geom = geometry
        self.pde = pde
        self.para_bd=para_bd
        self.para_ob=para_ob
        self.k_prox=k_prox
        self.BC_torch = bcs
        self.num_domain=num_domain
        self.observe_x=observe_x
        self.observe_u = observe_u
        
    def losses(self, model, loss_fn,random=False):
        if random:
            data_point=torch.from_numpy(self.geom.random_points(self.num_domain))
        else:
            data_point=torch.from_numpy(self.geom.uniform_points(self.num_domain))    
        x=data_point.to(torch.float32).to(device).to(device).requires_grad_(True)
        u= model(x)[0]
        k= model(x)[1]
        error_domain=self.pde(x,u,k,self.k_prox)
#         x_DBC=torch.from_numpy(self.geom.uniform_boundary_points()).to(torch.float32).to(device)
#         error_DBC=model(x_DBC)[0]-self.BC_torch(x_DBC)
        o_x=torch.from_numpy(self.observe_x).to(torch.float32).to(device)
        error_observe=model(o_x)[0]-torch.from_numpy(self.observe_u).to(torch.float32).to(device)
#         pred=torch.vstack((error_domain,self.para_bd*error_DBC,self.para_ob*error_observe))
        pred=torch.vstack((error_domain,self.para_ob*error_observe))
        loss = loss_fn(pred, torch.zeros_like(pred))
        return loss
"""
parameter
"""
beta=0.1
alpha=0.008/beta




"""
ADMM
"""
def k_ini(x):
    return x*0
k_iter=k_ini(x)
z_iter=k_iter.copy()
dual_iter=0*k_iter.copy()
for i_outer in range(50):
    "k subproblem"
    k_prox=z_iter+dual_iter/beta
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    data = IRCEE_V2(domain,pde,BC_torch,k_prox,num_domain=n,para_bd=1,para_ob=1,observe_x=x,observe_u=ud)
    epochs =20000
    for t in range(epochs):
#         print(f"Epoch {t+1}\n-------------------------------")
        train(model, loss_fn, optimizer)
    print(f"Done!{i_outer}")    
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=1000, max_eval=None, tolerance_grad=1e-09, tolerance_change=1e-09, history_size=100, line_search_fn= 'strong_wolfe' )
    epochs =1
    for t in range(epochs):
#         print(f"Epoch {t+1}\n-------------------------------")
        train_BFGS(model, loss_fn, optimizer)
#     print("Done!")
    "z subproblem"
    u=model(x_torch)[0]
    u_iter=u.cpu().detach().numpy()
    k=model(x_torch)[1]
    k_iter=k.cpu().detach().numpy()

    k_temp=k_iter-dual_iter/beta
   
    z_iter=TVdenoiser(k_temp, alpha)
    z_iter=np.where(z_iter>0.1,z_iter,0.1)
    plt.figure()
    plt.plot(x, k_iter, "-", label="k_o")
    plt.plot(x, z_iter, "--", label="k_denoised")
    plt.legend()
    plt.show()
    "dual update"
    dual_iter=dual_iter-beta*(k_iter-z_iter)

error=k_iter-ktrue
rel= np.linalg.norm(error)/np.linalg.norm(ktrue)    
print(f"relative error: {rel}")


# plt.figure()
# plt.plot(x, u_iter, "-", label="y")
# plt.plot(x,exact_u, "--", label="exact_y")
# plt.legend()
# plt.show()



# plt.figure()
# plt.plot(x, k_iter, "-", label="u")
# plt.plot(x,ktrue, "--", label="exact_u")
# plt.legend()
# plt.show()


error=k_iter-ktrue
rel= np.linalg.norm(error)/np.linalg.norm(ktrue)    
print(f"relative error: {rel}")  
u_fem= go.Scatter(
    x=x[:,0],
    y=u_iter[:,0],
    mode='markers+lines',
    name='y:ADMM-AtO-PINNs'
)
u_true= go.Scatter(
    x=x[:,0],
    y=exact_u[:,0],
    mode='lines',
    name='y:TRUE' 
)
k_fem= go.Scatter(
    x=x[:,0],
    y=k_iter[:,0],
    mode='markers+lines',
    name='u:ADMM-AtO-PINNs'
)
k_true= go.Scatter(
    x=x[:,0],
    y=ktrue[:,0],
    mode='lines',
    name='u:TRUE' 
)
# layout = go.Layout(legend={
#     'x':0.5,
#     'y':0.5
# })


error=k_iter-ktrue
rel= np.linalg.norm(error)/np.linalg.norm(ktrue)    
print(f"relative error: {rel}") 
fig = make_subplots(rows=1, cols=2,
                   subplot_titles=("recovered state", "recovered coefficient"))

fig.add_trace(u_fem,
              row=1, col=1)
fig.add_trace(u_true,
              row=1, col=1)
fig.add_trace(k_fem,
              row=1, col=2)
fig.add_trace(k_true,
              row=1, col=2)
fig.update_layout(height=400, width=900, title_text="Numerical results of ADMM-AtO-PINNs")
fig.show()
