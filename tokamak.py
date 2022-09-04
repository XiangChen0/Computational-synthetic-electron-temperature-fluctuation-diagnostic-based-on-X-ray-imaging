#import packages
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import interp2d


class tokamak():
    
    def __init__(self, qlist):
        self.qlist = qlist
         
    def fluxSurface(self, rho=0.5, r=None, theta_range=[0,2*np.pi], Ntheta=512):
        if r is None:
            assert rho>=0 and rho<=1
            r = np.interp(rho, self.qlist[0,:], self.qlist[1,:])
        R0 = np.interp(r, self.qlist[1,:], self.qlist[2,:])
        Z0 = np.interp(r, self.qlist[1,:], self.qlist[3,:])
        kappa = np.interp(r, self.qlist[1,:], self.qlist[4,:])
        delta = np.interp(r, self.qlist[1,:], self.qlist[5,:])
        zeta = np.interp(r, self.qlist[1,:], self.qlist[6,:])
        theta = np.linspace(theta_range[0], theta_range[1], num=Ntheta)
        R = R0 + r * np.cos(theta+np.arcsin(delta*np.sin(theta)))
        Z = Z0 + kappa * r * np.sin(theta+zeta*np.sin(2*theta))
            
        return R, Z  
    
    
    def plotFluxSurface(self, num_contours=20):      
        plt.rcParams.update({'font.size': 20})
        fig, axs = plt.subplots(1, 1, figsize=(8,6))
        for i in range(num_contours):
            R, Z =self.fluxSurface(rho=(i+1)/num_contours)
            axs.plot(R,Z)
        axs.set_aspect('equal','box')
        axs.set_xlabel('R/m')
        axs.set_ylabel('Z/m')
        plt.grid(linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
     
    #calculate the profiles ne(R,Z), Te(R,Z) over the whole cross section
    def profile(self, Nr=256, Ntheta=512):
        
        rmax = np.interp(1, self.qlist[0,:], self.qlist[1,:])
        r_glb = np.linspace(0,rmax,num=Nr)
        R_glb = np.zeros([Nr,Ntheta])
        Z_glb = np.zeros([Nr,Ntheta])
        ne_glb = np.zeros([Nr,Ntheta])
        Te_glb = np.zeros([Nr,Ntheta])
        for i in range(Nr):
            r = r_glb[i]
            R, Z = self.fluxSurface(r=r, Ntheta=Ntheta) 
            R_glb[i,:] = R
            Z_glb[i,:] = Z
            ne_glb[i,:] = np.interp(r, self.qlist[1,:], self.qlist[7,:])
            Te_glb[i,:] = np.interp(r, self.qlist[1,:], self.qlist[8,:]) 
            
        return R_glb, Z_glb, ne_glb, Te_glb
    
    
    #plot the profile
    def plotProfile(self, Nr=256, Ntheta=512):
        
        R_glb, Z_glb, ne_glb, Te_glb = self.profile(Nr=Nr, Ntheta=Ntheta)
        aspect = (np.max(Z_glb)-np.min(Z_glb))/(np.max(R_glb)-np.min(R_glb))
        
        plt.rcParams.update({'font.size': 20})
        fig, axs = plt.subplots(1,2, figsize=(np.round(7*2),7))
        c0 = axs[0].pcolor(R_glb, Z_glb, ne_glb, cmap='jet')
        fig.colorbar(c0, ax=axs[0],label='1e19 m-3')
        c1 = axs[1].pcolor(R_glb, Z_glb, Te_glb, cmap='jet')
        fig.colorbar(c1, ax=axs[1],label='keV')
        axs[0].set_xlabel('R/m')
        axs[0].set_ylabel('Z/m')
        axs[1].set_xlabel('R/m')
        axs[0].set_title('2D profile of electron density')
        #axs[1].set_title('2D profile of electron temp.')
        axs[0].grid(linestyle='--', linewidth=0.5)
        axs[1].grid(linestyle='--', linewidth=0.5)
        axs[0].set_xlim([0.3,1.55])
        axs[1].set_xlim([0.3,1.55])
        axs[0].set_ylim([-1.1,1.])
        axs[1].set_ylim([-1.1,1.])
        plt.tight_layout()
        plt.show()
        
        
    #calculate the X-ray emissivity given ne and Te; 
    #ne and Te have the same shape
    def emissivity(self, Ecutoff, ne, Te):
        
        return ne**2 * np.sqrt(Te) * np.exp(-Ecutoff/Te)  
        
        
    #calculate the intersections of a straight line Ax+By+C=0 and the outer surface of the plasma, the input is the 
    #coefficients specifying the straight line: line_params =[A,B,C]
    #the function returns the coordinates (R,Z,theta)
    def intersection(self, line_params, Ntheta = 1e5):
        
        R, Z = self.fluxSurface(rho=1, theta_range=[-np.pi,np.pi], Ntheta=int(Ntheta))
        dist = line_params[0]*R+line_params[1]*Z+line_params[2]
        inters = []
        for j in range(len(dist)-1):
            if dist[j]*dist[j+1]<=0:
                inters.append([(R[j]+R[j+1])/2,(Z[j]+Z[j+1])/2, (j+0.5)/Ntheta*2*np.pi-np.pi])
        #in case that the zeros are at the indices 0 or -1, but dist[j]*dist[j+1]<=0 is not satisfied
        #because of floating-point error    
        if len(inters) == 1: 
            inters = []
            R, Z = self.fluxSurface(rho=1, theta_range=[-np.pi/2,3/2*np.pi], Ntheta=int(Ntheta))
            dist = line_params[0]*R+line_params[1]*Z+line_params[2]
            for j in range(len(dist)-1):
                if dist[j]*dist[j+1]<=0:
                    inters.append([(R[j]+R[j+1])/2,(Z[j]+Z[j+1])/2, (j+0.5)/Ntheta*2*np.pi-np.pi/2])
                    
        return inters
    
    #(dr,dtheta) = J*(dR,dZ); r,theta are single numbers, not array-like
    def rth2RZ(self, r, theta): 
        
        rq = self.qlist[1,:]
        R0 = np.interp(r, rq, self.qlist[2,:])
        Z0 = np.interp(r, rq, self.qlist[3,:])
        kappa = np.interp(r, rq, self.qlist[4,:])
        delta = np.interp(r, rq, self.qlist[5,:])
        zeta = np.interp(r, rq, self.qlist[6,:])
        
        R = R0 + r * np.cos(theta+np.arcsin(delta*np.sin(theta)))
        Z = Z0 + kappa * r * np.sin(theta+zeta*np.sin(2*theta))
        
        return R, Z

    #(dr,dtheta) = J*(dR,dZ); r,theta are single numbers, not array-like
    def Jacobian(self, r, theta): 
        
        r = np.array(r).reshape(-1,)
        theta = np.array(theta).reshape(-1,)
        
        rq = self.qlist[1,:]
        kappa = np.interp(r, rq, self.qlist[4,:])
        delta = np.interp(r, rq, self.qlist[5,:])
        zeta = np.interp(r, rq, self.qlist[6,:]) 
        dR0dr = np.interp(r, rq, self.qlist[9,:])
        dZ0dr = np.interp(r, rq, self.qlist[10,:])
        dkdr = np.interp(r, rq, self.qlist[11,:])
        dddr = np.interp(r, rq, self.qlist[12,:])
        dzdr = np.interp(r, rq, self.qlist[13,:])
        theta1 = theta + np.arcsin(delta*np.sin(theta))
        darc = 1/np.sqrt(1-(delta*np.sin(theta))**2)
        theta2 = theta + zeta * np.sin(2*theta)
        #(dR,dZ) = Jinv*(dr,dtheta)  
        dRdr = dR0dr + np.cos(theta1) - dddr*r*np.sin(theta1)*np.sin(theta)*darc
        dRdth = -r * np.sin(theta1) * (1+delta*np.cos(theta)*darc)
        dZdr = dZ0dr + (r*dkdr+kappa)*np.sin(theta2) + dzdr*kappa*r*np.cos(theta2)*np.sin(2*theta)
        dZdth = kappa*r*np.cos(theta2)*(1+2*zeta*np.cos(2*theta))
        det = dRdr*dZdth - dRdth*dZdr
        
        J = np.zeros([2,2,len(r)])
        J[0,0,:] = dZdth/det
        J[0,1,:] = -dRdth/det
        J[1,0,:] = -dZdr/det
        J[1,1,:] = dRdr/det
        
        return np.squeeze(J)
    
    #given (R,Z), find (r,theta) through gradient descent method
    #fast to solve (r, theta) for a single point (R,Z)
    # L = (R(r,theta)-Rtarget)^2 + (Z(r,theta)-Ztarget)^2
    # dL/dr = 2*(R-Rtarget)*dRdr + 2*(Z-Ztarget)*dZdr
    # dL/dth = 2*(R-Rtarget)*dRdth + 2*(Z-Ztarget)*dZdth
    # (r,theta) -= lr*(dL/dr, dL/dth) = lr*J*(R-Rtarget, Z-Ztarget)
    
    def RZ2rth(self, R, Z, start_point=None, error_tol=1e-8, lr=1e-2):
        
        if start_point is None:
            r, theta = np.random.rand(), 2*np.pi*np.random.rand()
        else:
            r, theta = start_point[0], start_point[1]
            
        error = 1
        
        while error > error_tol:
            Rtmp, Ztmp = self.rth2RZ(r, theta)
            error = (Rtmp-R)**2 + (Ztmp-Z)**2
            J = np.linalg.inv(self.Jacobian(r,theta))
            r = r - lr * ((Rtmp-R)*J[0,0]+(Ztmp-Z)*J[1,0])
            theta = theta - lr*((Rtmp-R)*J[0,1]+(Ztmp-Z)*J[1,1])
            if r < 0:
                r, theta = np.random.rand(), 2*np.pi*np.random.rand()
                  
        return r, theta%(2*np.pi)



    def RZmeshgrid(self, Rgrid=np.linspace(0.2,1.6,num=1000), Zgrid=np.linspace(-1.2, 1.2, num=1000)):
        
        dR, dZ = Rgrid[1]-Rgrid[0], Zgrid[1]-Zgrid[0]
        Rv, Zv = np.meshgrid(Rgrid, Zgrid)
        rv, thv = np.zeros(Rv.shape), np.zeros(Rv.shape)
        
        #upward (Z-increasing) calculation           
        rv[0,0], thv[0,0] = self.RZ2rth(Rv[0,0], Zv[0,0], start_point=[1, np.pi*5/4])
        for i in range(Rv.shape[1]-1):
            J = self.Jacobian(rv[0,i],thv[0,i])
            rv[0,i+1], thv[0,i+1] = rv[0,i]+J[0,0]*dR, thv[0,i]+J[1,0]*dR  
        for j in range(Rv.shape[0]-1):
            J = self.Jacobian(rv[j,:],thv[j,:])
            rv[j+1,:], thv[j+1,:] = rv[j,:]+J[0,1,:]*dZ, thv[j,:]+J[1,1,:]*dZ  
        
        rv1, thv1 = rv.copy(), thv.copy()
        rv1[rv<0] = np.nan
        thv1[rv<0] = np.nan
        
        #downward (Z-decreasing) calculation
        rv[-1,0], thv[-1,0] = self.RZ2rth(Rv[-1,0], Zv[-1,0], start_point=[1, np.pi*3/4])
        for i in range(Rv.shape[1]-1):
            J = self.Jacobian(rv[-1,i],thv[-1,i])
            rv[-1,i+1], thv[-1,i+1] = rv[-1,i]+J[0,0]*dR, thv[-1,i]+J[1,0]*dR
        for j in range(Rv.shape[0]-1,0,-1):
            J = self.Jacobian(rv[j,:],thv[j,:])
            rv[j-1,:], thv[j-1,:] = rv[j,:]-J[0,1,:]*dZ, thv[j,:]-J[1,1,:]*dZ 
            
        thv[rv<0] = np.nan
        rv[rv<0] = np.nan
        
        import warnings
        warnings.filterwarnings("ignore")
        
        rv = np.nanmean(np.stack([rv1,rv], axis=2), axis=2)
        thv = np.nanmean(np.stack([thv1%(2*np.pi),thv%(2*np.pi)], axis=2), axis=2)
        
        #remove the remaining NANs
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]
        
        Rs = np.unique(Rv[np.isnan(rv)])
        
        for R in Rs:
            y1 = rv[Rv==R]
            nans, x = nan_helper(y1)
            y1[nans] = np.interp(x(nans), x(~nans), y1[~nans])
            rv[Rv==R] = y1
            
            y2 = thv[Rv==R]
            nans, x = nan_helper(y2)
            y2[nans] = np.interp(x(nans), x(~nans), y2[~nans])
            thv[Rv==R] = y2          
        
        return Rv, Zv, rv, thv

        
    def interp_emis(self, energy_threshold, time=100, RZgrid=None, real_fluct=True, test_fluct=False):
        if RZgrid is None:
            Rgrid = np.linspace(0.2,1.6,num=2000)
            Zgrid = np.linspace(-1.2, 1.2, num=2000) 
        else:
            Rgrid = np.linspace(0.2,1.6,num=RZgrid[0])
            Zgrid = np.linspace(-1.2, 1.2, num=RZgrid[1])
            
        Rv, Zv, rv, thv = self.RZmeshgrid(Rgrid, Zgrid) #shape:(N_Zgrid, N_Rgrid)
        
        dne_glb, dTe_glb = self.fluct(time=time, grid=[512,1024])
        R_glb, Z_glb, ne_glb, Te_glb = self.profile(Nr=dne_glb.shape[0], Ntheta=dne_glb.shape[1])
        
        if real_fluct:
            ems_glb = self.emissivity(energy_threshold, ne_glb+ne_glb*dne_glb/100, Te_glb+Te_glb*dTe_glb/100)
        else:
            ems_glb = self.emissivity(energy_threshold, ne_glb, Te_glb) #(Nr, Ntheta)
        
        rmax = np.interp(1, self.qlist[0,:], self.qlist[1,:])
        
        r_glb = np.linspace(0, rmax, num=ems_glb.shape[0])
        th_glb = np.linspace(0, 2*np.pi, num=ems_glb.shape[1])
        emis_rth = RBS(r_glb, th_glb, ems_glb, kx=3, ky=3)
        
        emisv = emis_rth.ev(rv, thv) #(NZgrid, NRgrid)
        
        if ~real_fluct and test_fluct:
            emisv = emisv * (1 + 0.05 * np.cos(400*Rv)) #test fluctuations
            
        emisv[rv>rmax] = 0
        emis_RZ = RBS(Rgrid, Zgrid, emisv.T, kx=3, ky=3)
        
        return emis_RZ
        
    # calculate the line integration of X-ray emissivity epsilon
    def brightness(self, line_params, interp_spline, RZrange=[0.2,1.6,-1.2,1.2], num_sample=1e3):
        
        try: 
            if line_params[0] == 0:
                chord_R = np.linspace(RZrange[0],RZrange[1],num=int(num_sample))
                chord_Z = -line_params[2]/line_params[1]*np.ones(chord_R.shape)           
            elif line_params[1] == 0:
                chord_Z = np.linspace(RZrange[2],RZrange[3],num=int(num_sample))
                chord_R = -line_params[2]/line_params[0]*np.ones(chord_Z.shape) 
            else:
                R1 = -(line_params[1]*RZrange[2]+line_params[2])/line_params[0]
                R2 = -(line_params[1]*RZrange[3]+line_params[2])/line_params[0]
                Rs = sorted(RZrange[:2]+[R1,R2])
                Zi = -(line_params[0]*Rs[1]+line_params[2])/line_params[1]
                Zf = -(line_params[0]*Rs[2]+line_params[2])/line_params[1]
                chord_R = np.linspace(Rs[1],Rs[2],num=int(num_sample))
                chord_Z = np.linspace(Zi,Zf,num=int(num_sample))

            dl = np.linalg.norm([chord_R[1]-chord_R[0], chord_Z[1]-chord_Z[0]])
            emis = interp_spline.ev(chord_R, chord_Z)
            bright = np.trapz(emis, dx=dl) #(np.sum(emis[:-1])+np.sum(emis[1:]))/2*dl
            
        except:
            bright = 0
           
        return bright
    
#-----------------------------------------------------------------------------------------------------------
# (!!! inefficient) discarded codes: calculate the coordinate (r,theta) along the LOS one by one from (R,Z) 
#-----------------------------------------------------------------------------------------------------------
#             #(dR, dZ)
#             dR_dZ = np.array([chord_R[1]-chord_R[0], chord_Z[1]-chord_Z[0]]).reshape((2,1))
#             #initial values of (r, theta)
#             r_theta = np.array([self.qlist[1,-1], intersections[0][-1]]).reshape((2,1)) 
#             rthetas = np.zeros([2,len(chord_R)])
#             rthetas[:,0] = r_theta.reshape((2,))           
#             #for i in range(len(chord_R)-1):
#             i = 0
#             flag = 0
#             while i < len(chord_R)-1:
#                 # the Jacobian matrix is singular at r=0, i.e. when the LOS passes through the point r=0, to prevent
#                 # weird things from happening, the strategy is to jump to the other end of the LOS and integrate the
#                 # other way around
#                 if flag == 0 and abs(rthetas[0,i]) < 1e-3:
#                     flag = 1
#                     stop = i
#                     i = len(chord_R)-1
#                     r_theta = np.array([self.qlist[1,-1], intersections[1][-1]]).reshape((2,1)) 
#                     rthetas[:,-1] = r_theta.reshape((2,))
                    
#                 if flag == 0:
#                     J = self.Jacobian(rthetas[0,i],rthetas[1,i])
#                     r_theta += np.dot(J,dR_dZ)
#                     rthetas[:,i+1] = r_theta.reshape((2,))
#                     i += 1
#                 else:
#                     J = self.Jacobian(rthetas[0,i],rthetas[1,i])
#                     r_theta -= np.dot(J,dR_dZ)
#                     rthetas[:,i-1] = r_theta.reshape((2,))
#                     i -= 1
#                     if i == stop:
#                         break


    
    def fluct(self, time=100, grid=None, rhoc_lis=[0.5, 0.6, 0.7], lovera_lis = [0.9687, 0.7590, 0.6391]):
        
        #'grid' can be input in the format of the shape of a numpy array, e.x. [256,512]
        if grid is None: 
            dne = np.genfromtxt(f'dne_{time}.csv', delimiter=',')
            grid = dne.shape
        rho_glb = np.linspace(0,1,num=grid[0])
        theta_glb = np.linspace(0, 2*np.pi, num=grid[1])
        dne_glb = np.zeros(grid) #shape: grid
        dTe_glb = np.zeros(grid)
        
        for j, rhoc in enumerate(rhoc_lis[:1]):
            dir_workspace = f'./'
            lovera = lovera_lis[j]
            dne = np.genfromtxt(dir_workspace+f'dne_{time}.csv', delimiter=',')
            dTe = np.genfromtxt(dir_workspace+f'dTe_{time}.csv', delimiter=',')           
            rho_loc = np.linspace(rhoc-0.5*lovera, rhoc+0.5*lovera, num=dne.shape[0])
            theta_loc = np.linspace(0, 2*np.pi, num=dne.shape[1])
            
            b = np.logical_and(rho_loc<=1, rho_loc>=0)
            rnorm = (rho_loc-rhoc)/lovera
            #rnorm[:] = 0
            scaler = np.exp(4*rnorm**3)
#             if j==0: #(rhoc == 0.5)
#                 scaler[rho_loc<rhoc] = np.exp(-rnorm[rho_loc<rhoc]**2/2/8)
            scaler[~b] = 0
            scaler = scaler.reshape(-1,1)
            
            interp_dne = RBS(rho_loc, theta_loc, dne*scaler)
            interp_dTe = RBS(rho_loc, theta_loc, dTe*scaler)
            ind = np.logical_and(rho_glb >= min(rho_loc), rho_glb <= max(rho_loc))
            rho_intp, theta_intp = np.meshgrid(rho_glb[ind], theta_glb, indexing='ij') #shape:(grid[0][ind], grid[1])
            dne_glb[ind,:] += interp_dne.ev(rho_intp,theta_intp)
            dTe_glb[ind,:] += interp_dTe.ev(rho_intp,theta_intp)
                
        return dne_glb, dTe_glb
    
    
    def plotFluct(self):
        
        dne_glb, dTe_glb = self.fluct(grid=[512,1024])
        #print(dne_glb.shape)
        R_glb, Z_glb, ne_glb, Te_glb = self.profile(Nr=dne_glb.shape[0], Ntheta=dne_glb.shape[1])
        #print(np.sum(np.isnan(R_glb)), np.sum(np.isnan(dne_glb)), np.sum(np.isnan(ne_glb)))
        
        aspect = (np.max(Z_glb)-np.min(Z_glb))/(np.max(R_glb)-np.min(R_glb))
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(3,2,figsize=(10,np.round(10*aspect)))
        
        c0 = ax[0,0].pcolor(R_glb,Z_glb,dne_glb, cmap='jet')
        fig.colorbar(c0, ax=ax[0,0])
        c1 = ax[0,1].pcolor(R_glb,Z_glb,dTe_glb, cmap='jet')
        fig.colorbar(c1, ax=ax[0,1])
        c2 = ax[1,0].pcolor(R_glb,Z_glb,ne_glb*dne_glb/100, cmap='jet')
        fig.colorbar(c2, ax=ax[1,0], label='1e19 m-3')
        c3 = ax[1,1].pcolor(R_glb,Z_glb,Te_glb*dTe_glb/100, cmap='jet')
        fig.colorbar(c3, ax=ax[1,1], label='keV')
        c4 = ax[2,0].contourf(R_glb,Z_glb,ne_glb+ne_glb*dne_glb/100, cmap='jet', levels=20)
        fig.colorbar(c4, ax=ax[2,0], label='1e19 m-3')
        c5 = ax[2,1].contourf(R_glb,Z_glb,Te_glb+Te_glb*dTe_glb/100,cmap='jet', levels=20)
        fig.colorbar(c5, ax=ax[2,1], label='keV')
        
        ax[0,0].set_title('density fluct. level(%)')
        ax[0,1].set_title('temp. fluct. level(%)')
        ax[1,0].set_title('density fluct.')
        ax[1,1].set_title('temp. fluct.')
        ax[2,0].set_title('density profile + fluct.')
        ax[2,1].set_title('temp. profile + fluct.')
        ax[0,0].set_ylabel('Z/m')
        ax[1,0].set_ylabel('Z/m')
        ax[2,0].set_ylabel('Z/m')
        ax[2,0].set_xlabel('R/m')
        ax[2,1].set_xlabel('R/m')
        for ele in ax.flatten():
            ele.grid(linestyle='--', linewidth=0.5)
            
        plt.tight_layout()
        plt.show()

        
if __name__ == "__main__":
    
    qlist = np.genfromtxt('qlist.csv', delimiter=',')
    
    NSTX = tokamak(qlist)

# ------------------------------
# test the function "fluxSurface"
# ------------------------------
    NSTX.plotFluxSurface()

# ------------------------------
# test the function "profile"
# ------------------------------
# NSTX.plotProfile()

# ------------------------------
# test the function "fluct"
# ------------------------------
# NSTX.plotFluct()

# ------------------------------
# test the function "RZ2rth"
# ------------------------------
# for R in np.linspace(0.2,2,num=40):
#     r, th = NSTX.RZ2rth(R,-0.5)
#     print(r, th)
#     #print(NSTX.rth2RZ(r, th))
#     print(NSTX.Jacobian(r, th))
#-1.2506709939809546, 0.6001962410874857

# ------------------------------
# test the function "RZmeshgrid"
# ------------------------------
# Rv, Zv, rv, thv = NSTX.RZmeshgrid()
# print(rv.shape)
#emis_RZ = NSTX.interp_emis(energy_threshold=0.5)


# ------------------------------
# test the function "Jacobian"
# ------------------------------
# J = NSTX.Jacobian(1, -2.862713473730627)
# print(J)
# J = NSTX.Jacobian([0.1,0.2,0.3], [np.pi/2,np.pi/3,np.pi/4])
# print(J)

# ------------------------------
# test the function "intersection"
# ------------------------------
# z0 = NSTX.qlist[3,-1]
# print(z0)
# intersections = NSTX.intersection([0,1,-z0]) #straight line y=z0
# print(intersections)
# dR1 = intersections[1][0]-intersections[0][0]
# print(dR1)
# R_glb, Z_glb, ne_glb, Te_glb = NSTX.profile()
# dR2 = np.max(R_glb)-np.min(R_glb)
# print(dR2) #if |dR1|=|dR2|, the function "intersection" is functioning

# ------------------------------
# test the function "brightness"
# ------------------------------
# R_glb, Z_glb, ne_glb, Te_glb = NSTX.profile()
# dne_glb, dTe_glb = NSTX.fluct()
# ems = NSTX.emissivity(0.5, ne_glb, Te_glb)
# r_glb = np.linspace(0, NSTX.qlist[1,-1], num=256)
# th_glb = np.linspace(-np.pi, np.pi, num=512)
# ne_glb[:] = 1

# interp_spline = RBS(r_glb, th_glb, ne_glb)
# #print(interp_spline.ev([0.3,0.25], [-1,2]))

# z0 = NSTX.qlist[3,-1]
# bt = NSTX.brightness([0,1,-z0], interp_spline)
# print(bt, np.max(R_glb)-np.min(R_glb))

# plt.pcolor(R_glb, Z_glb, ems, cmap='jet')
# plt.tight_layout()
# plt.show()