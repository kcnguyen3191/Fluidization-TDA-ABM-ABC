import numpy as np
import pandas as pd
import concurrent.futures
from scipy.integrate import ode
import matplotlib.pyplot as plt
import os
import glob
import imageio as io
from itertools import repeat

class Dorsogna_fluidization:
    def __init__(self,sigma=0.00224,alpha=0.01,beta=0.1,
                 cA=0.02,cR=0.04,lA=0.12,lR=0.06,
                 BCs=('left','right','top','bottom')):
        ###Model parameters
        #Random walk diffusion parameter [mm/hr^.5]
        self.sigma = sigma
        #Self-propulsion parameter [cell-mass/hr]
        self.alpha = alpha
        #Friction parameter [cell-mass*hr/mm^2]
        self.beta = beta
        #Attractive strength [cell-mass*mm^2/hr^2] "work"
        self.cA = cA
        #Repulsive strength [cell-mass*mm^2/hr^2] "work"
        self.cR = cR
        #Attractive radius [mm]
        self.lA = lA
        #Repulsive radius [mm]
        self.lR = lR
        #Create parameter dict for easy reference
        self.params = {'sigma':sigma,'alpha':alpha,'beta':beta,
                       'cA':cA,'cR':cR,'lA':lA,'lR':lR}
        #Include parameter priors to be used for bayes, bounds, etc
        self.priors = {'sigma':(0.0,0.002),
                       'alpha':(0.001,0.1),
                       'beta':(0.01,0.5),
                       'cA':(0.0001,0.006),
                       'cR':(0.0005,0.008),
                       'lA':(0.005,0.03),
                       'lR':(0.001,0.02)}

        ###Boundary conditions
        if not set(BCs).issubset({'left','right','top','bottom'}):
            raise ValueError("Invalid BC in {0}".format(BCs))
        self.BCs = BCs

    def diff(self,t,Z):
        '''
        Gets derivatives of position and velocity of particles according to the
        D'Orsogna model of soft-core particle interactions.
        Inputs:
            t: (unused) time, for integrator use only
            Z: (1-d ndarray, 4*num_cells long) current step position and velocity
        Output: derivative of position and velocity at current step to be
        integrated to obtain position and velocity at next step
        '''
        meps = np.finfo(np.float64).eps
        #Get ICs from input vector
        num_cells = len(Z)//4
        x = Z[0:num_cells][None,:]
        y = Z[num_cells:2*num_cells][None,:]
        vx = Z[2*num_cells:3*num_cells]
        vy = Z[3*num_cells:]

        #Compute model components
        xdiff = x - x.T
        ydiff = y - y.T
        D = np.sqrt(xdiff**2+ydiff**2)
        with np.errstate(over='raise'):
            v_normSq = vx**2 + vy**2
        u_prime = - self.cA/self.lA*np.exp(-D/self.lA) +\
                  self.cR/self.lR*np.exp(-D/self.lR)

        #Calculate model terms and return as ndarray
        dvxdt = (self.alpha - self.beta*v_normSq)*vx -\
                np.sum(u_prime*xdiff/(D+meps),axis=1)
        dvydt = (self.alpha - self.beta*v_normSq)*vy -\
                np.sum(u_prime*ydiff/(D+meps),axis=1)

        #Incorporate boundaries
        x = np.squeeze(x)
        y = np.squeeze(y)
        # if 'left' in self.BCs:
        #     vx[x<-0.05] = np.abs(vx[x<-0.05])
        # if 'right' in self.BCs:
        #     vx[x>1.212] = -np.abs(vx[x>1.212])
        # if 'top' in self.BCs:
        #     vy[y<-0.05] = np.abs(vy[y<-0.05])
        # if 'bottom' in self.BCs:
        #     vy[y>0.935] = -np.abs(vy[y>0.935])
        if 'left' in self.BCs:
            vx[x<0] = np.abs(vx[x<0])
        if 'right' in self.BCs:
            vx[x>1.16162489196] = -np.abs(vx[x>1.16162489196])
        if 'top' in self.BCs:
            vy[y<0] = np.abs(vy[y<0])
        if 'bottom' in self.BCs:
            vy[y>0.88504753673] = -np.abs(vy[y>0.88504753673])

        output = np.hstack((vx,vy,dvxdt,dvydt))
        return output

    def ode_rk4(self,ic_vec,t0,tf,
                dt=1/6):
        '''
        Simulate position and velocity until last desired frame using RK4/5.
        Inputs:
            ic_vec: (ndarray) vector of concatenated initial position and velocity
            t0: (float) time of ic_vec
            tf: (float) final time
        Output:
            simu: (list of ndarrays) list of simulated cells at each frame
        Kwargs:
            df: (float) time step for results
        '''
        simu = [ic_vec]
        r = ode(self.diff).set_integrator('dopri5',atol=10**(-3))
        r.set_initial_value(ic_vec,t0)
        while r.successful() and r.t < tf:
            #print("\rSimulating time {0:.3f}".format(r.t+dt),end='')
            simu.append(r.integrate(r.t+dt))
        #print("\n",end='')

        #Set class attribute for use in other methods
        self.simulation_results = simu

    def _dw(self,num_cells,dt):
        #Get Weiner process for SDE definition
        dw = np.random.normal(loc=0.0,scale=np.sqrt(dt),size=(num_cells*4,))
        return dw

    def sde_maruyama(self,ic_vec,t0,tf,dt=1/600,
                     return_time=1/6):
        '''
        Simulate SDE with Brownian motion in position using Euler-Maruyama
        until last desired frame.
        Inputs:
            ic_vec: (ndarray) vector of concatenated initial position and velocity
            t0: (float) time of ic_vec
            tf: (float) final time
        Kwargs:
            dt: (float) time step for simulation (default 1/600)
            return_time: (float) how often to return values (default 1/6)
        Output:
            simu: (list of ndarrays) list of simulated cells at each frame
        '''
        num_cells = len(ic_vec)//4
        simu = [ic_vec]
        #Create vector of time steps for simulation and return
        return_vec = np.arange(t0,tf+return_time,return_time)
        time_vec = np.union1d(return_vec,np.arange(t0,tf+dt,dt))
        #Create vector for stochastic term of SDE
        b = np.hstack((np.full((num_cells,),self.sigma,dtype=np.float64),
                       np.full((num_cells,),self.sigma,dtype=np.float64),
                       np.zeros((num_cells,),dtype=np.float64),
                       np.zeros((num_cells,),dtype=np.float64)))
        #Simulate until end
        for j in time_vec:
            #if j in return_vec:
                #print("\rSimulating time {0:.3f}".format(j),end='')
            ns = simu[-1] + self.diff(j,simu[-1])*dt + b*self._dw(num_cells,dt)
            simu.append(ns)
        #print("\n",end='')
        #Return only desired time steps
        simu = [s for (s,j) in zip(simu,time_vec) if j in return_vec]
        #Set class attribute for use in other methods
        self.simulation_results = simu

    def position_gif(self,figure_dir,time_vec):
        '''Create GIF of simulated cell positions over time.
        Inputs:
            figure_dir: (str) directory in which to save image(s)
            time_vec: (list/ndarray) list of frame numbers for titling
        '''
        images = []
        #Make scatter plot of each frame in time_vec
        for j,(t,vec) in enumerate(zip(time_vec,self.simulation_results)):
            plt.plot(vec[0:len(vec)//4],vec[len(vec)//4:len(vec)//2],'b.',ms=3)
            plt.title("Time: {0:.3f} hrs".format(t,time_vec[-1]))
            plt.xlim(0,1.16162489196)
            plt.xlabel(r"$x\ [mm]$")
            plt.ylim(0.88504753673,0)
            plt.ylabel(r"$y\ [mm]$")
            save_path = os.path.join(figure_dir,"frame_"+str(j)+".png")
            plt.savefig(save_path)
            plt.close()
            images.append(io.imread(save_path))

        #Combine frames into GIF and delete individual plots
        io.mimsave(os.path.join(figure_dir,"position.gif"),images,fps=8)
        for png_path in glob.glob(os.path.join(figure_dir,"*.png")):
            os.remove(png_path)

    def results_to_df(self,time_vec):
        num_cells = len(self.simulation_results[0])//4
        particles = range(num_cells)
        df = pd.DataFrame(columns=['t','x','y','vx','vy','particle'])
        for j,(vec,t) in enumerate(zip(self.simulation_results,time_vec)):
            frame = pd.DataFrame(
                {'t':t,'x':vec[:num_cells],'y':vec[num_cells:2*num_cells],
                 'vx':vec[2*num_cells:3*num_cells],'vy':vec[3*num_cells:],
                 'particle':np.arange(num_cells,dtype=np.uint16),'frame':j})
            df = df.append(frame,ignore_index=True)

        df = df.astype({'frame':np.uint16})
        return df
