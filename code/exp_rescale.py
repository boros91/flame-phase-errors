"""Front delay of RK45 for the flame problem in the original variables (t,y) and in the
linearly rescaled variables s = delta*t, z = y/delta (Paper A, Section on time as a dependent variable)."""
import numpy as np
rows=[]
from scipy.integrate import solve_ivp
F=lambda y: 1/y+np.log(1/y-1)          # t = F(delta) - F(y)
for d in [1e-3,1e-4,1e-5,1e-6]:
    for tol in [1e-6]:
        # original, relative + small absolute tol
        s=solve_ivp(lambda t,y: y**2-y**3,(0,2/d),[d],method='RK45',rtol=tol,atol=1e-3*tol*d,dense_output=True)
        y=s.y[0]; t=s.t; ok=(y>d*(1+1e-12))&(y<0.99); k=np.argmax(y[ok]>=0.5)
        D0=t[ok][k]-(F(d)-F(y[ok][k]))
        # rescaled z=y/d, tau=d t, to tau=2
        r=solve_ivp(lambda tau,z: z**2*(1-d*z),(0,2),[1.0],method='RK45',rtol=tol,atol=1e-3*tol,dense_output=True)
        yz=d*r.y[0]; tt=r.t/d; ok=(yz>d*(1+1e-12))&(yz<0.99); k=np.argmax(yz[ok]>=0.5)
        D1=tt[ok][k]-(F(d)-F(yz[ok][k]))
        rows.append((d,D0,D1)); print(f"delta={d:.0e} tol={tol:.0e}  front delay original={D0:+.3e}  rescaled={D1:+.3e}  (front width ~13)")
import csv; csv.writer(open("../results/rescale.csv","w")).writerows([("delta","D_original","D_rescaled")]+rows)
