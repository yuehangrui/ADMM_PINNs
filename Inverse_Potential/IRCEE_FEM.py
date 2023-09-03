import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
# plotly.offline.init_notebook_mode()
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.fftpack import fft 
from scipy.fftpack import ifft
np.random.seed(0)
"""
geom
"""
x_range=(0,1)
N=100
h=1/N
xp = np.arange(x_range[0], x_range[1]+h, h)
x= np.zeros([N+1,1])

x[:,0]=xp
"""
true data
"""

def k(x):
    return 0.2 + 0.8*np.float32((x > 0.25)*(x<0.75))

ktrue = k(x)
nu=0.005

f= np.sin(2 * np.pi * x)

# f= nu*4*np.pi**2*np.sin(2 * np.pi * x)+k(x)*np.sin(2 * np.pi * x)
e=np.ones(N+1)*nu
diag_data=np.array([2*e+ktrue[:,0]*h**2,-e,-e])
A=spdiags(diag_data,[0,1,-1],N+1,N+1)
A=A.tocsr()
A[0,0]=1
A[0,1]=0
A[1,0]=0
A[-1,-1]=1
A[-2,-1]=0
A[-1,-2]=0
A=A.todense()
exact_u=np.linalg.solve(A,f*h**2)
"""
ud
"""
noise_sigma=0.05*np.linalg.norm(exact_u)*h
noise=np.random.randn(N+1,1)*noise_sigma
ud=exact_u+noise
# plt.figure()
# plt.plot(x, exact_u, "-", label="u_true")
# plt.plot(x, ud, "*-", label="u_noised")
# plt.legend()
# plt.show()
"""
A0
"""
diag_data0=np.array([2*e,-e,-e])
A0=spdiags(diag_data0,[0,1,-1],N+1,N+1)
A0=A0.todense()
A0[0,0]=1
A0[0,1]=0
A0[1,0]=0
A0[-1,-1]=1
A0[-2,-1]=0
A0[-1,-2]=0
"""
ini
"""
def k_ini(x):
    return 0*x
k_iter=k_ini(x)
A_iter=A0+np.diag(k_iter[:,0])*h**2
A_iter[0,0]=1
A_iter[-1,-1]=1
u_iter=np.linalg.solve(A_iter,f*h**2)
z_iter=k_iter
dual_iter=0.*k_iter
"""
ADMM
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
"""
parameter
"""
beta=0.1
alpha=0.008/beta
for i_outer in range(50):
    k_prox=z_iter+dual_iter/beta
    """
    Gauss Newton
    """
    for i_GN in range(50):
        U_iter=np.diag(u_iter[:,0])*h**2
        inv_A=np.linalg.inv(A_iter)
        v_iter=inv_A.dot(u_iter-ud)*h**2
        gradient=beta*(k_iter-k_prox)-np.multiply(u_iter,v_iter)
        graident_norm=np.linalg.norm(gradient*h)
        # print(f"graident norm : {graident_norm}")
        obj=beta*np.linalg.norm(k_iter-k_prox)**2+np.linalg.norm(u_iter-ud)**2
        # print(f"obj: {obj}")
        DS=-inv_A.dot(U_iter)
        GN=DS.dot(DS.transpose())+beta*np.identity(N+1)
        Delta_k=-np.linalg.solve(GN,gradient)
        rho=1
        for i_testsize in range(5):
            k_test=k_iter.copy()
            k_test=k_test+rho*Delta_k
            A_test=A0+np.diag(k_test[:,0])*h**2
            A_test[0,0]=1
            A_test[-1,-1]=1
            u_test=np.linalg.solve(A_test,f*h**2)
            v_test=inv_A.dot(u_test-ud)*h**2
            obj_test=beta*np.linalg.norm(k_test-k_prox)**2+np.linalg.norm(u_test-ud)**2
            gradient_test=beta*(k_test-k_prox)-np.multiply(u_test,v_test)
            graident_norm_test=np.linalg.norm(gradient_test*h)
            if obj_test<obj:
                continue
            else:
                rho=rho*0.9
        if  graident_norm<1e-5:
            continue
        k_iter=k_iter+rho*Delta_k
        k_iter=np.where(k_iter>0.1,k_iter,0.1)
        A_iter=A0+np.diag(k_iter[:,0])*h**2
        A_iter[0,0]=1
        A_iter[-1,-1]=1
        u_iter=np.linalg.solve(A_iter,f*h**2)
    k_temp=k_iter-dual_iter/beta
    k_temp=np.where(k_temp>0.1,k_temp,0.1)
    z_iter=TVdenoiser(k_temp, alpha)
    # plt.figure()
    # plt.plot(x, k_iter, "-", label="k_o")
    # plt.plot(x, z_iter, "--", label="k_denoised")
    # plt.legend()
    # plt.show()
    dual_iter=dual_iter-beta*(k_iter-z_iter)


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
u_fem= go.Scatter(
    x=x[:,0],
    y=u_iter[:,0],
    mode='markers+lines',
    name='y:ADMM-FEM'
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
    name='u:ADMM-FEM'
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
fig.update_layout(height=400, width=900, title_text="Numerical results of ADMM-FEM")
fig.show()
