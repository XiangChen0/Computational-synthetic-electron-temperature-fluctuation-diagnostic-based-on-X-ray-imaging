# %load XrayDiagnostic.py

#import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp2d
from scipy.integrate import simpson
from datetime import datetime
import multiprocessing as mp
from scipy.interpolate import RectBivariateSpline as RBS
from sklearn.linear_model import LinearRegression
#ridge regression
from sklearn.linear_model import Ridge
from scipy.ndimage import gaussian_filter
from tokamak import *


class XrayDiagnostic(tokamak):
    
    def __init__(self, qlist, geometry, num_glbLOS, num_locLOS, num_angles, t_wall, roi, extension_factor, beam_size, SNR, sample_rate):
        super().__init__(qlist)
        self.geometry= geometry
        self.num_glbLOS = num_glbLOS #number of global light-of-sights(LOS) per Xray camera
        self.num_locLOS = num_locLOS #number of local light-of-sights(LOS) per Xray camera
        self.num_angles = num_angles #number of viewing angles where X-ray cameras are installed 
        self.t_wall = t_wall #thickness of the first wall or the distance from the aperture of the camera to the plasma edge
        self.beam_size = beam_size #beam size of LOS
        self.roi = roi #region of interest, roi = [[R_min, Rmax],[Z_min, Z_max]]
        self.extension_factor = extension_factor
        
        self.SNR = SNR #signal-to-noise ratio of X-ray detectors 
        self.sample_rate = sample_rate #sampling rate of X-ray detectors
        
    def camera(self, Zedge, height2angle=False):
        
        R, Z = self.fluxSurface(rho=1, Ntheta=1024)
        
        Redge = []
        tvecedge = []
        for i in range(len(Z)-1):
            if (Z[i]-Zedge)*(Z[i+1]-Zedge)<0 or Z[i]==Zedge:
                Redge.append((R[i]+R[i+1])/2)
                tvecedge.append([R[i+1]-R[i],Z[i+1]-Z[i]])
        tvecedge = tvecedge[0] if Redge[0]>=Redge[1] else tvecedge[1]
        Redge = max(Redge)
        nvecedge = [tvecedge[1],-tvecedge[0]]/np.linalg.norm(tvecedge)
        Rapert = Redge + self.t_wall * nvecedge[0]
        Zapert = Zedge + self.t_wall * nvecedge[1] #location of aperture (Rapert, Zapert)
        
        if height2angle: # only calculate the line parameters of the center LOS passing through the center of ROI
            Rc, Zc = np.mean(self.roi[0]), np.mean(self.roi[1])
            p1 = Zc - Zapert
            p2 = Rapert - Rc
            theta = np.arctan2(p2, p1)
            #theta = theta + np.pi * (theta<0)
            return theta 
            
        
        #find the tangent vector (dR,dZ) at each point (R,Z) on the plasma edge
        dR = np.zeros(R.shape)
        dR[1:-1] = R[2:] - R[:-2]
        dR[0] = R[1] - R[-2]
        dR[-1] = dR[0]
        dZ = np.zeros(Z.shape)
        dZ[1:-1] = Z[2:] - Z[:-2]
        dZ[0] = Z[1] - Z[-2]
        dZ[-1] = dZ[0]

        #(Rtangent, Ztangent) is the tangential point of the tangent of (Rapert, Zapert) and the plasma edge profile, 
        #there should be 2 such points
        Rtangent = []
        Ztangent = []
        cprod = dR*(Z-Zapert)-dZ*(R-Rapert)
        for i in range(len(cprod)-1):
            if cprod[i]*cprod[i+1]<0 or cprod[i]==0:
                Rtangent.append((R[i]+R[i+1])/2)
                Ztangent.append((Z[i]+Z[i+1])/2)
                
        #global chords        
        if Ztangent[0] <= Ztangent[1]:
            Ztangent.reverse()
            Rtangent.reverse()
  
        cosTheta = ((Rtangent[0]-Rapert)*(Rtangent[1]-Rapert)+(Ztangent[0]-Zapert)*(Ztangent[1]-Zapert))/np.linalg.norm([Rtangent[0]-Rapert, Ztangent[0]-Zapert])/np.linalg.norm([Rtangent[1]-Rapert, Ztangent[1]-Zapert])
        Theta = np.arccos(cosTheta)
        glbtheta = np.linspace(0, Theta, num=self.num_glbLOS+2)
        line_params = []
        for th in glbtheta[1:-1]:
            p1 = (Ztangent[0]-Zapert)*np.cos(th) + (Rtangent[0]-Rapert)*np.sin(th)
            p2 = (Ztangent[0]-Zapert)*np.sin(th) - (Rtangent[0]-Rapert)*np.cos(th)
            p3 = -p1*Rapert-p2*Zapert
            line_params.append([p1,p2,p3])
            
        #local chords
        hs = self.extension_factor * max(self.roi[0][1]-self.roi[0][0], self.roi[1][1]-self.roi[1][0])/2
        s = np.linspace(-hs, hs, num=self.num_locLOS)
        Rc, Zc = np.mean(self.roi[0]), np.mean(self.roi[1])
        Rs = Rc + (Zapert-Zc)*s/np.sqrt((Rapert-Rc)**2+(Zapert-Zc)**2)
        Zs = Zc - (Rapert-Rc)*s/np.sqrt((Rapert-Rc)**2+(Zapert-Zc)**2)
        
        if self.geometry == 'fanbeam':#fan beam geometry
            p1 = Zs - Zapert
            p2 = Rapert - Rs
            p3 = (Rs-Rapert)*Zapert - (Zs-Zapert)*Rapert
        elif self.geometry == 'parallel':#parallel geometry 
            p1 = (Zapert - Zc) * np.ones_like(s)
            p2 = (Rc - Rapert) * np.ones_like(s)
            p3 = (Rapert-Rc)*Zs + (Zc-Zapert)*Rs
        
        line_params = line_params + np.vstack([p1,p2,p3]).T.tolist()

            
        return line_params
    
    
    def camera_heights_for_even_angles(self):
    
        heights = np.linspace(-0.95, 0.95, num=1000)

        angles = [self.camera(height, height2angle=True) for height in heights]
        
        heights_pos = [height for angle,height in zip(angles,heights) if angle>0]
        
        angles_pos = [angle for angle,height in zip(angles,heights) if angle>0]

        angles_query = np.linspace(angles_pos[0], angles_pos[-1], num=self.num_angles)

        heights_query = np.interp(angles_query, angles_pos, heights_pos)
        
        #print([self.camera(height, height2angle=True)/np.pi for height in heights_query])

        return heights_query
    
    
        
    def plotDiag(self, camera_heights, camera_size=[0.2,0.1]):
        
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(1,1,figsize=(8,10))
        R, Z = self.fluxSurface(rho=1, Ntheta=1024)

        
        for ch in camera_heights:
            
            line_params = self.camera(ch)
            reg = LinearRegression().fit(np.array(line_params)[:, :2], -np.array(line_params)[:, -1])
            camera_location = reg.coef_
            
            #plot cameras
            dxy = camera_size[0]/2*np.array(line_params[self.num_glbLOS//2][:2])/np.linalg.norm(line_params[self.num_glbLOS//2][:2])
            rotation_angle = 360 - np.arccos(-dxy[0]/np.linalg.norm(dxy))/np.pi*180
            #print(camera_location, rotation_angle)
            rect = patches.Rectangle(np.array(camera_location)+dxy, camera_size[0], camera_size[1], angle=rotation_angle, 
                                     linewidth=1, edgecolor='y', facecolor='y')
            ax.add_patch(rect)
            
            #plot LOS(line of sight)
            count = 0
            for param in line_params:
                intersections = self.intersection(param)
                bd = np.linalg.norm(intersections[0][:2]-camera_location) < np.linalg.norm(intersections[1][:2]-camera_location)
                if abs(param[0]) > abs(param[1]):
                    y = np.linspace(intersections[int(bd)][1],camera_location[1],num=100)
                    x = -(param[-1]+param[1]*y)/param[0]
                else:
                    x = np.linspace(intersections[int(bd)][0],camera_location[0],num=100)  
                    y = -(param[-1]+param[0]*x)/param[1]
                if count < self.num_glbLOS:
                    ax.plot(x, y, color='r', linewidth=2)
                else:
                    ax.plot(x, y, color='g', linewidth=1)
                count += 1
                
                
        #plot region of interest
        #ax.plot(self.roi, [0,0], linewidth=8, color='fuchsia')
        rect_roi = patches.Rectangle((self.roi[0][0],self.roi[1][0]), self.roi[0][1]-self.roi[0][0], self.roi[1][1]-self.roi[1][0],
                                     linewidth=8, edgecolor='b', facecolor='b', zorder=2)
        ax.add_patch(rect_roi)
        #last flux surface
        ax.plot(R,Z,'k')
        
        #plt.xlim([min(np.min(R), camera_location[0])-0.2,max(np.max(R), camera_location[0])+0.2])
        ax.axis('equal')
        plt.xlim([0, 2])
        plt.ylim([min(np.min(Z), camera_location[1])-0.2,max(np.max(Z), camera_location[1])+0.2])
        plt.xlabel('R/m')
        plt.ylabel('Z/m')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('Xraydiag_config.png')
        plt.show()
        
    
    def measurement(self, time=100, Eths=[0.5,1.0,1.5], camera_heights=[-0.9,-0.5,-0.1,0.1,0.5,0.9], RZgrid=[1000,1000], real_fluct=True, test_fluct=False, num_sample_per_LOS=1024, projection_space=False):
        
        #print(time, mp.current_process())
        
        interp_splines = [self.interp_emis(Eth, time=time, RZgrid=RZgrid, real_fluct=real_fluct, test_fluct=test_fluct) for Eth in Eths]
        
        ROIcenter = [np.mean(self.roi[0]), np.mean(self.roi[1])]
        
        if projection_space:
            
            tot_s, tot_theta, tot_meas = [], [], []

            for camera_height in camera_heights:

                line_params= self.camera(camera_height)

                meas = [[self.brightness(lp, sp, num_sample=num_sample_per_LOS) for lp in line_params] for sp in interp_splines]

                s, theta = self.projection(ROIcenter, line_params)

                tot_s.append(s)

                tot_theta.append(theta)

                tot_meas.append(meas)
                
            return np.array(tot_s), np.array(tot_theta), np.array(tot_meas) #shape(Nth, Nlos) for tot_s
        
        else:
            
            tot_meas = []

            for camera_height in camera_heights:

                line_params = self.camera(camera_height)

                meas = [[self.brightness(lp, sp, num_sample=num_sample_per_LOS) for lp in line_params] for sp in interp_splines]

                tot_meas.append(meas)
            
            return np.array(tot_meas) #(Nth, Nenergy, Nlos)
    
    
    
    #global reconstruction
    def pixelize(self, Rlim=[0.3,1.5], Zlim=[-1.1,1.0], RZshape=[100,100]):
        #pixelize the whole plasma region inside the tokamak and 
        #return a flattened lists of the center of each pixel
        Rrange = np.linspace(Rlim[0], Rlim[1], num=RZshape[0])
        Zrange = np.linspace(Zlim[0], Zlim[1], num=RZshape[1])
        
        dR = Rrange[1] - Rrange[0]
        dZ = Zrange[1] - Zrange[0]    
        
        Rc, Zc, rc, _ = self.RZmeshgrid(Rgrid=(Rrange[1:]+Rrange[:-1])/2, Zgrid=(Zrange[1:]+Zrange[:-1])/2)
        
        qs = self.qlist
        rout = qs[1,qs[0,:]==1][0]
        
        Rcin = Rc[rc < rout]
        Zcin = Zc[rc < rout]
        
        return Rcin, Zcin, dR, dZ, Rc, Zc, rc<rout
    
    
    def constructX(self, Rc, Zc, dR, dZ, line_params):
        
        Rem, Rep, Zem, Zep = Rc-dR/2, Rc+dR/2, Zc-dZ/2, Zc+dZ/2
        
        sign1 = np.sign(line_params[0]*Rem+line_params[1]*Zem+line_params[2])
        sign2 = np.sign(line_params[0]*Rem+line_params[1]*Zep+line_params[2])
        sign3 = np.sign(line_params[0]*Rep+line_params[1]*Zem+line_params[2])
        sign4 = np.sign(line_params[0]*Rep+line_params[1]*Zep+line_params[2])
        
        sign_sum = sign1 + sign2 + sign3 + sign4
        
        inds = np.where(np.abs(sign_sum)<4) #the straight line intersects with the rectangle centered at [Rc, Zc]
        #Rw = np.vstack([Rc[inds], Rep[inds], Zc[inds], Rem[inds]])
        #print(Rw.shape)
        #dc = np.abs(line_params[0]*Rc+line_params[1]*Zc+line_params[2]) / np.linalg.norm(line_params[:-1])
        #dmax = np.sqrt(dR**2+dZ**2)
        #ind = np.where(dc<dmax)
        
        X = np.zeros(Rc.shape)
        
        if line_params[0] == 0:
            X[inds] = dR
        elif line_params[1] == 0:
            X[inds] = dZ
        else:
            Rm = -(line_params[1]*Zem[inds]+line_params[2])/line_params[0]
            Rp = -(line_params[1]*Zep[inds]+line_params[2])/line_params[0]
            Rs = np.sort(np.vstack([Rem[inds], Rep[inds], Rm, Rp]), axis=0)
            X[inds] = np.abs( np.linalg.norm(line_params[:-1]) / line_params[1] * (Rs[1,:]-Rs[2,:]))
            
        return X
            
        
    @staticmethod    
    def emis2temp(emis, Eths = [0.5,1.0,1.5]): #emis: (n_samples, n_energy)
        
        from scipy.optimize import curve_fit
        
        def func(x, a, b, c):
            return a * np.exp(-b*x) + c

        te = np.zeros((emis.shape[0],)) #shape: (n_samples, )
        
        for i in range(len(te)):
            try:
                popt, pcov = curve_fit(func, Eths, emis[i,:], bounds=([0,0.5,-np.inf], np.inf))
                te[i] = 1/popt[1]
            except:
                pass
            
        return te
            
            
    def global_reconstruction(self, camera_heights, totmeas, Eths = [0.5,1.0,1.5], ifplot=False):
        
        
        Rcin, Zcin, dR, dZ, Rc, Zc, cdin = self.pixelize(Rlim=[0.3,1.6], Zlim=[-1.1,1.0], RZshape=[101,101])
        X = Rcin
        
        for height in camera_heights:
            line_params = self.camera(height) 
            for line_param in line_params[:self.num_glbLOS]:
                X0 = self.constructX(Rcin, Zcin, dR, dZ, line_param)
                X = np.vstack([X, X0])
        X = X[1:,:]
        
        ems = None
        spline_ems = []
        for i in range(totmeas.shape[1]):
            y = np.squeeze(totmeas[:,i,:self.num_glbLOS]).reshape(-1,)
            clf = Ridge(alpha=1e-4, fit_intercept=False, solver='lbfgs', positive=True).fit(X, y)
            em = np.zeros(Rc.shape)
            em[cdin] = clf.coef_
            em_smoothed = gaussian_filter(em, sigma=4)
            spline_em = RBS(Rc[0,:], Zc[:,0], em_smoothed.T)
            spline_ems.append(spline_em)
            ems = em_smoothed[cdin] if ems is None else np.vstack([ems, em_smoothed[cdin]])
        
        if ifplot:
            
            te = np.zeros(Rc.shape)
            te[cdin] = self.emis2temp(ems.T, Eths)
            
            te_smoothed = gaussian_filter(te, sigma=4)

            spline_te = RBS(Rc[0,:], Zc[:,0], te_smoothed.T)
            
            te[~cdin] = np.nan
            te_smoothed[~cdin] = np.nan  
            
            #aspect = (np.max(Zc)-np.min(Zc))/(np.max(Rc)-np.min(Rc))
            plt.rcParams.update({'font.size': 20})
            fig, axs = plt.subplots(1,2, figsize=(np.round(7*2),7))
            c0 = axs[0].pcolor(Rc, Zc, te, cmap='jet')
            fig.colorbar(c0, ax=axs[0],label='keV')
            c1 = axs[1].pcolor(Rc, Zc, te_smoothed, cmap='jet')
            fig.colorbar(c1, ax=axs[1],label='keV')
            axs[0].set_xlabel('R/m')
            axs[0].set_ylabel('Z/m')
            axs[1].set_xlabel('R/m')
            #axs[0].set_title('2D profile of reconstructed electron temp.')
            #axs[1].set_title('2D profile of reconstructed electron temp.(smoothed)')
            axs[0].grid(linestyle='--', linewidth=0.5)
            axs[1].grid(linestyle='--', linewidth=0.5)
            axs[0].set_xlim([0.3,1.55])
            axs[1].set_xlim([0.3,1.55])
            axs[0].set_ylim([-1.1,1.])
            axs[1].set_ylim([-1.1,1.])
            plt.tight_layout()
            plt.show()
            
            return spline_ems, spline_te
        
        return spline_ems, None
    
    
    @staticmethod
    def projection(ROIcenter, line_params):
        
        ss = []
        thetas = []
        for lp in line_params:
            
            s = (lp[0]*ROIcenter[0]+lp[1]*ROIcenter[1]+lp[2]) / np.linalg.norm(lp[:2])
            theta = np.arctan2(-lp[1]*s, -lp[0]*s) #-pi~pi #np.pi/2 if lp[0]==0 else np.arctan(lp[1]/lp[0])
            ss.append(np.abs(s)*np.sign(theta))
            thetas.append((theta+np.pi)%(np.pi))
        
        return ss,thetas
    
    
    def local_reconstruction(self, totss, totthetas, totmeas, query_point_size=[20,20], interp_method='linear'):

        #xqs: query points
#         if MP:
#             sloc = totss[0][self.num_glbLOS:]
#             #Rqs = sloc
#             Rqs = sloc[np.abs(sloc)<= (self.roi[0][1]-self.roi[0][0])/2]
#             Zqs = np.array([0]) 
#         else:
            
        Rqs = np.linspace(self.roi[0][0], self.roi[0][1], num=query_point_size[0]) - np.mean(self.roi[0]) if query_point_size[0]>1 else np.array([0]) 
        Zqs = np.linspace(self.roi[1][0], self.roi[1][1], num=query_point_size[1]) - np.mean(self.roi[1]) if query_point_size[1]>1 else np.array([0]) 
            
        xqs = np.meshgrid(Rqs, Zqs, indexing='ij') #shape: (n_R, n_Z)
            
        if self.geometry == 'fanbeam':
            #projection space (s，theta), where pi>theta>0. There are discontinuities at theta=0 and pi. 
            #For the purpose of interpolation, we need to do continuation to expand the range of theta to (-pi, 2*pi).
            from scipy.interpolate import griddata
            
            sexp = np.vstack([-totss[:, self.num_glbLOS:], totss[:, self.num_glbLOS:],-totss[:, self.num_glbLOS:]])

            thexp = np.vstack([totthetas[:, self.num_glbLOS:]-np.pi, totthetas[:, self.num_glbLOS:], totthetas[:, self.num_glbLOS:]+np.pi])

            measexp = np.vstack([totmeas[:,:, self.num_glbLOS:], totmeas[:,:, self.num_glbLOS:], totmeas[:,:, self.num_glbLOS:]])

            points = np.vstack([sexp.flatten(),thexp.flatten()]).T

            grid_s, grid_th = np.mgrid[np.min(sexp)*0.95:np.max(sexp)*0.95:60j, 0:np.pi:80j]

            n_s, n_th = grid_s.shape

            n_energy = totmeas.shape[1]

            ems = None

            for j in range(n_energy):

                values = measexp[:,j,:].flatten()

                grid_meas = griddata(points, values, (grid_s, grid_th), method=interp_method)

                freq = 2*np.pi*np.fft.fftfreq(n_s, d=grid_s[1,0]-grid_s[0,0]) #shape(n_s, )

                spectra = np.fft.fft(grid_meas, axis=0) #shape(n_s, n_th)

                intgrd = np.fft.ifft(spectra*np.abs(freq.reshape(-1,1)), axis=0) #shape(n_s, n_th)

                fths = [np.interp(xqs[0].flatten()*np.cos(grid_th[0,i])+xqs[1].flatten()*np.sin(grid_th[0,i]), 
                        grid_s[:,i], intgrd[:,i]) for i in range(n_th)] 

                fths = np.array(fths).T #shape(n_xq, n_th)

                em = np.trapz(fths, x=grid_th[0,:], axis=-1)/2/np.pi #shape(n_xq, )
                
                em = em.reshape(xqs[0].shape)

                ems = em if ems is None else np.dstack((ems, em))

            return ems, xqs
        
        
        elif self.geometry == 'parallel':
            #parallel geometry(regular grids in projection space, no need for interpolation,
            #but need to sort the points in order)
            
            grid_s = totss[:, self.num_glbLOS:].T.copy() #shape:(n_s, n_th)

            grid_th = totthetas[:, self.num_glbLOS:].T.copy() #shape:(n_s, n_th)

            grid_meas = np.transpose(totmeas[:,:, self.num_glbLOS:], (1, 2, 0)).copy() #shape:(n_energy, n_s, n_th)

            n_energy, n_s, n_th = grid_meas.shape

            #sort by s in the ascending order    
            ind_s = np.argsort(grid_s, axis=0) 
            grid_s = np.take_along_axis(grid_s, ind_s, axis=0)
            grid_th = np.take_along_axis(grid_th, ind_s, axis=0)
            #sort by the angles in the ascending order
            ind_th = np.argsort(grid_th, axis=1)
            grid_s = np.take_along_axis(grid_s, ind_th, axis=1)
            grid_th = np.take_along_axis(grid_th, ind_th, axis=1)

            ems = None

            for j in range(n_energy):

                freq = np.fft.fftfreq(n_s, d=grid_s[1,0]-grid_s[0,0]) #shape(n_s, )

                grid_meas[j,:,:] = np.take_along_axis(grid_meas[j,:,:], ind_s, axis=0) #sort measurements by s and theta
                grid_meas[j,:,:] = np.take_along_axis(grid_meas[j,:,:], ind_th, axis=1)
                
                grid_meas[j,:,:] = np.array([grid_meas[j,:,k] - np.polyval(np.polyfit(grid_s[:,k], grid_meas[j,:,k], 5), grid_s[:,k]) for k in range(n_th)]).T

                spectra = np.fft.fft(grid_meas[j,:,:], axis=0) #shape(n_s, n_th)

                intgrd = np.fft.ifft(spectra*np.abs(freq).reshape(-1,1), axis=0) #shape(n_s, n_th)

                fths = [np.interp(xqs[0].flatten('F')*np.cos(grid_th[0,i])+xqs[1].flatten('F')*np.sin(grid_th[0,i]), 
                        grid_s[:,i], intgrd[:,i]) for i in range(n_th)]

                fths = np.array(fths).T #shape(n_xq, n_th)

                em = simpson(fths, x=grid_th[0,:], axis=-1) #shape(n_xq, )

                em = em.reshape(xqs[0].shape, order='F')

                ems = em if ems is None else np.dstack((ems, em))

            return ems, xqs
        
        
    def reconstrct_Te(self, tot_meas=None, tot_s=None, tot_theta=None, camera_heights=[-0.9,-0.5,0.5,0.9], Eths=[0.5,1.0,1.5], grid_measure_points=[80,20], addout=False):
        
        em_local, xqs = self.local_reconstruction(tot_s, tot_theta, tot_meas, query_point_size=grid_measure_points)

        em_local = em_local.real
        
        spline_ems, _ = self.global_reconstruction(camera_heights, tot_meas, ifplot=False)
        
        em_global = np.array([spline_em.ev(xqs[0]+np.mean(self.roi[0]), xqs[1]+np.mean(self.roi[1])) for spline_em in spline_ems])
        
        em_global = np.transpose(em_global, (1, 2, 0))
        
        nr, nz, ne = em_global.shape
        
        Te = np.zeros([nr, nz])
        
        for i in range(nr):
            
            em_tot = em_global[i,:,:] + em_local[i,:,:]
            #em_tot = em_local[i,:,:]
            
            Te[i,:] = self.emis2temp(em_tot, Eths)
            
            #print(np.min(em_tot))

        if addout:
            return Te, xqs, em_global, em_local
        
        return Te

################################    
    
    def local_reconstruction(self, totss, totthetas, totmeas, query_point_size=[20,20], interp_method='linear'):
        
        
            
        Rqs = np.linspace(self.roi[0][0], self.roi[0][1], num=query_point_size[0]) - np.mean(self.roi[0]) if query_point_size[0]>1 else np.array([0]) 
        Zqs = np.linspace(self.roi[1][0], self.roi[1][1], num=query_point_size[1]) - np.mean(self.roi[1]) if query_point_size[1]>1 else np.array([0]) 
            
        xqs = np.meshgrid(Rqs, Zqs, indexing='ij') #shape: (n_R, n_Z)
            
        if self.geometry == 'fanbeam':
            #projection space (s，theta), where pi>theta>0. There are discontinuities at theta=0 and pi. 
            #For the purpose of interpolation, we need to do continuation to expand the range of theta to (-pi, 2*pi).
            from scipy.interpolate import griddata
            
            sexp = np.vstack([-totss[:, self.num_glbLOS:], totss[:, self.num_glbLOS:],-totss[:, self.num_glbLOS:]])

            thexp = np.vstack([totthetas[:, self.num_glbLOS:]-np.pi, totthetas[:, self.num_glbLOS:], totthetas[:, self.num_glbLOS:]+np.pi])

            measexp = np.vstack([totmeas[:,:, self.num_glbLOS:], totmeas[:,:, self.num_glbLOS:], totmeas[:,:, self.num_glbLOS:]])

            points = np.vstack([sexp.flatten(),thexp.flatten()]).T

            grid_s, grid_th = np.mgrid[np.min(sexp)*0.95:np.max(sexp)*0.95:60j, 0:np.pi:80j]

            n_s, n_th = grid_s.shape

            n_energy = totmeas.shape[1]

            ems = None

            for j in range(n_energy):

                values = measexp[:,j,:].flatten()

                grid_meas = griddata(points, values, (grid_s, grid_th), method=interp_method)

                freq = 2*np.pi*np.fft.fftfreq(n_s, d=grid_s[1,0]-grid_s[0,0]) #shape(n_s, )

                spectra = np.fft.fft(grid_meas, axis=0) #shape(n_s, n_th)

                intgrd = np.fft.ifft(spectra*np.abs(freq.reshape(-1,1)), axis=0) #shape(n_s, n_th)

                fths = [np.interp(xqs[0].flatten()*np.cos(grid_th[0,i])+xqs[1].flatten()*np.sin(grid_th[0,i]), 
                        grid_s[:,i], intgrd[:,i]) for i in range(n_th)] 

                fths = np.array(fths).T #shape(n_xq, n_th)

                em = np.trapz(fths, x=grid_th[0,:], axis=-1)/2/np.pi #shape(n_xq, )
                
                em = em.reshape(xqs[0].shape)

                ems = em if ems is None else np.dstack((ems, em))

            return ems, xqs
        
        
        elif self.geometry == 'parallel':
            #parallel geometry(regular grids in projection space, no need for interpolation,
            #but need to sort the points in order)
            
            grid_s = totss[:, self.num_glbLOS:].T.copy() #shape:(n_s, n_th)

            grid_th = totthetas[:, self.num_glbLOS:].T.copy() #shape:(n_s, n_th)

            grid_meas = np.transpose(totmeas[:,:, self.num_glbLOS:], (1, 2, 0)).copy() #shape:(n_energy, n_s, n_th)

            n_energy, n_s, n_th = grid_meas.shape

            #sort by s in the ascending order    
            ind_s = np.argsort(grid_s, axis=0) 
            grid_s = np.take_along_axis(grid_s, ind_s, axis=0)
            grid_th = np.take_along_axis(grid_th, ind_s, axis=0)
            #sort by the angles in the ascending order
            ind_th = np.argsort(grid_th, axis=1)
            grid_s = np.take_along_axis(grid_s, ind_th, axis=1)
            grid_th = np.take_along_axis(grid_th, ind_th, axis=1)

            ems = None

            for j in range(n_energy):

                freq = np.fft.fftfreq(n_s, d=grid_s[1,0]-grid_s[0,0]) #shape(n_s, )

                grid_meas[j,:,:] = np.take_along_axis(grid_meas[j,:,:], ind_s, axis=0) #sort measurements by s and theta
                grid_meas[j,:,:] = np.take_along_axis(grid_meas[j,:,:], ind_th, axis=1)
                
                grid_meas[j,:,:] = np.array([grid_meas[j,:,k] - np.polyval(np.polyfit(grid_s[:,k], grid_meas[j,:,k], 5), grid_s[:,k]) for k in range(n_th)]).T

                spectra = np.fft.fft(grid_meas[j,:,:], axis=0) #shape(n_s, n_th)

                intgrd = np.fft.ifft(spectra*np.abs(freq).reshape(-1,1), axis=0) #shape(n_s, n_th)

                fths = [np.interp(xqs[0].flatten('F')*np.cos(grid_th[0,i])+xqs[1].flatten('F')*np.sin(grid_th[0,i]), 
                        grid_s[:,i], intgrd[:,i]) for i in range(n_th)]

                fths = np.array(fths).T #shape(n_xq, n_th)

                em = simpson(fths, x=grid_th[0,:], axis=-1) #shape(n_xq, )

                em = em.reshape(xqs[0].shape, order='F')

                ems = em if ems is None else np.dstack((ems, em))

            return ems, xqs
        
        
    def reconstrct_Te(self, tot_meas=None, tot_s=None, tot_theta=None, camera_heights=[-0.9,-0.5,0.5,0.9], Eths=[0.5,1.0,1.5], grid_measure_points=[80,20], addout=False):
        
        em_local, xqs = self.local_reconstruction(tot_s, tot_theta, tot_meas, query_point_size=grid_measure_points)

        em_local = em_local.real
        
        spline_ems, _ = self.global_reconstruction(camera_heights, tot_meas, ifplot=False)
        
        em_global = np.array([spline_em.ev(xqs[0]+np.mean(self.roi[0]), xqs[1]+np.mean(self.roi[1])) for spline_em in spline_ems])
        
        em_global = np.transpose(em_global, (1, 2, 0))
        
        nr, nz, ne = em_global.shape
        
        Te = np.zeros([nr, nz])
        
        for i in range(nr):
            
            em_tot = em_global[i,:,:] + em_local[i,:,:]
            #em_tot = em_local[i,:,:]
            
            Te[i,:] = self.emis2temp(em_tot, Eths)
            
            #print(np.min(em_tot))

        if addout:
            return Te, xqs, em_global, em_local
        
        return Te

#----------------------------------
# test the function "camera"
#----------------------------------
# NSTXdiag = XrayDiagnostic(qlist, num_glbLOS=10, num_locLOS=20, num_angles=6, t_wall=0.2, roi=[1.25,1.35], beam_size=0.01, SNR=100, sample_rate=2)

# x = np.linspace(0,2,num=100)

# fig, ax = plt.subplots(1,1,figsize=(8,10))

# camera_locations = [-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9]

# for height in camera_locations:
    
#     line_params = NSTXdiag.camera(Zedge=height)   
#     for line_param in line_params:
#         y = (-line_param[2]-line_param[0]*x)/line_param[1]
#         ax.plot(x,y,'k')
    
# NSTX = tokamak(qlist)
# R,Z = NSTX.fluxSurface(rho=1, theta_range=[-np.pi,np.pi], Ntheta=512)
# ax.plot(R,Z,'r')
# ax.plot(1.041354,-2.3134e-2,'ro')
# plt.tight_layout()
# plt.xlim([0.2,1.7])
# plt.ylim([-1.2,1.2])
# plt.grid()
# plt.show()

