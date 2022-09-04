from datetime import datetime
import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS
from sklearn.linear_model import LinearRegression
import multiprocessing as mp
from functools import partial
from tokamak import *
from XrayDiagnostic import *


if __name__ == "__main__":
    
    tic = datetime.now()

    qlist = np.genfromtxt('qlist.csv', delimiter=',')

    NSTX = tokamak(qlist)

    geometry = 'parallel'
    num_glbLOS, num_locLOS, num_angles = 20, 80, 10
    extension_factor = 4
    t_wall = 0.2
    roi = [[1.25, 1.3],[0.0, 0.001]]
    beam_size = 0.01
    SNR = 100
    sample_rate = 2

    NSTXdiag = XrayDiagnostic(qlist, geometry=geometry, num_glbLOS=num_glbLOS, num_locLOS=num_locLOS, num_angles=num_angles, 
                              t_wall=t_wall, roi=roi, extension_factor=extension_factor, beam_size=beam_size,
                              SNR=SNR, sample_rate=sample_rate)

    camera_heights = NSTXdiag.camera_heights_for_even_angles()
    #camera_heights = np.linspace(-0.95, 0.95, num=num_angles) 
    #camera_heights = [-0.9, -0.6, -0.5, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.5, 0.6, 0.9]
    #camera_heights = [-0.9, -0.5, -0.2, 0, 0.2, 0.5, 0.9]

    Eths = [0.5, 1.0, 1.5]

    tot_s, tot_theta, tot_meas_nf = NSTXdiag.measurement(time=100, Eths=Eths, camera_heights=camera_heights, RZgrid=[1000,1000],
                                                         real_fluct=False, test_fluct=False, num_sample_per_LOS=2000, projection_space=True)
    
    Te_nf, xqs, _, _ = NSTXdiag.reconstrct_Te(tot_meas_nf, tot_s, tot_theta, camera_heights, Eths, grid_measure_points=[80,20], addout=True)
    
    ncpu = mp.cpu_count()
    print(ncpu)
    pool_obj = mp.Pool(ncpu)
    
    measure_at_time = partial(NSTXdiag.measurement, Eths=Eths, camera_heights=camera_heights, RZgrid=[1000,1000],
                                                     real_fluct=True, test_fluct=False, num_sample_per_LOS=2000, projection_space=False)
    times = np.arange(100,301)
    
    tot_meas_wfs = pool_obj.map(measure_at_time,times)

    toc = datetime.now()
    print(f'elapsed time is {toc-tic}')
    tic = datetime.now()
    
    reconst_at_meas = partial(NSTXdiag.reconstrct_Te, tot_s=tot_s, tot_theta=tot_theta, camera_heights=camera_heights,
                     Eths=Eths, grid_measure_points=[80,20], addout=False)

    Te_wfs = pool_obj.map(reconst_at_meas, tot_meas_wfs)

    tot_meas_wfs = np.array(tot_meas_wfs)

    Te_wfs = np.array(Te_wfs)

    print(tot_meas_wfs.shape, Te_wfs.shape)

    np.savetxt("tot_meas_wfs.csv", tot_meas_wfs.flatten(), delimiter=",")

    toc = datetime.now()
    print(f'elapsed time is {toc-tic}')
    tic = datetime.now()

    Te = np.mean(Te_wfs, axis=-1)
    dTe = Te/np.mean(Te,axis=0)-1

    ndeg = 8
    X = np.arange(0,dTe.shape[-1])
    dTe_denoise = dTe.copy()
    for i in range(dTe.shape[0]):
        pcoefs = np.polyfit(X, dTe[i,:], ndeg)
        dTe_pred = np.polyval(pcoefs, X)
        dTe_denoise[i,:] = dTe[i,:] - dTe_pred
    
    print(dTe.shape, dTe_denoise.shape)
    
    #Rgrid=np.linspace(0.2,1.6,num=1000)
    #Zgrid=np.linspace(-1.2, 1.2, num=1000)
    #Rv, Zv, rv, thv = NSTX.RZmeshgrid(Rgrid=Rgrid, Zgrid=Zgrid)

    #spline_r = RBS(Rgrid, Zgrid, rv.T)
    #spline_th = RBS(Rgrid, Zgrid, thv.T)

    #Rq, Zq = xqs[0]+np.mean(roi[0]), xqs[1]+np.mean(roi[1]) #(n_R, n_Z)

    #rq = spline_r.ev(Rq, Zq)
    #thq = spline_th.ev(Rq, Zq)

    #dTe_gt_tot = []
    #dTe_gt_cor_tot = []
    #for i in range(0,len(times)):
    #    if i%20==0:
    #        print(i)
        
    #    dne_glb, dTe_glb = NSTX.fluct(time=times[i], grid=[512,1024])
    #    spline = RBS(np.linspace(0,1,num=512), np.linspace(0, 2*np.pi, num=1024), dTe_glb/100) #shape:(n_r, n_th)
    #    dTe_gt = spline.ev(rq, thq)
    #    dTe_gt_tot.append(np.mean(dTe_gt,axis=-1))

    #    ndeg = 8
    #    for i in range(dTe_gt.shape[1]):
    #        y = dTe_gt[:,i]
    #        X = np.arange(0,len(y))
    #        pcoefs = np.polyfit(X, y, ndeg)
    #        y_pred = np.polyval(pcoefs, X)
    #        dTe_gt[:,i] = y - y_pred
        
    #    dTe_gt_cor_tot.append(np.mean(dTe_gt,axis=-1))
    
    np.savetxt("Te_wfs.csv",Te_wfs.flatten(),delimiter=",")
    np.savetxt("dTe.csv",dTe.flatten(),delimiter=",")
    np.savetxt("dTe_denoised.csv",dTe_denoise.flatten(),delimiter=",")
    #np.savetxt("dTe_gt.csv",np.array(dTe_gt_tot).flatten(),delimiter=",")
    #np.savetxt("dTe_gt_denoised.csv",np.array(dTe_gt_cor_tot).flatten(),delimiter=",")

    fig, ax = plt.subplots(1,2,figsize=(16,5))
    ax[0].contourf(dTe)
    ax[1].contourf(dTe_denoised)
    plt.savefig('dTe.png')

    fig, ax = plt.subplots(1,2,figsize=(16,5))
    sp = np.abs(np.fft.fftshift(np.fft.fft2(dTe)))
    sp_denoised = np.abs(np.fft.fftshift(np.fft.fft2(dTe_denoised)))
    ax[0].contourf(sp)
    ax[1].contourf(sp_denoised)
    plt.savefig('spectra.png')

    toc = datetime.now()
    print(f'elapsed time is {toc-tic}')
