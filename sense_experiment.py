""" Preliminaries """
from bart import bart
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
import cfl
from multiprocessing import Pool
from time import time
import os
from glob import glob
import h5py


def disp(im, ttl='',prcntmx = 99, figsize=(10,3),plt_phase=False,subplt=(1,1,1)): # to avoid code repetition
    if subplt[2]==1:
        plt.figure(figsize=figsize)
    ax = plt.subplot(*subplt)
    ax.set_xticks([0, im.shape[0]-1])
    ax.set_yticks([0 , im.shape[1]-1])
    if plt_phase:
        myfun = np.angle
        v = None
    else:
        myfun = np.abs
        v = np.percentile(myfun(im),prcntmx)
    ax.imshow(myfun(np.squeeze(im)), cmap='gray', vmax=v)
    plt.title(ttl) 

# upat
def gen_upat_2d(ny,nz,R,ncent,do_z_usamp=None):
    # returns 2d uniform sampling pattern of ny x nz, filled in center of centy x centz, else skipping w/ dy, dz
    assert ny%2==0 and nz%2==0, 'only even dimensions supported'
    do_z_usamp = nz<ny if do_z_usamp is None else do_z_usamp
    upat = np.zeros([ny,nz])
    if do_z_usamp:
        dy,dz=1,R
        centy,centz=ny,ncent
    else:
        dy,dz=R,1  
        centy,centz=ncent,ny
    idx_skipy,idx_skipz = np.int_(np.r_[0:ny:dy]),np.int_(np.r_[0:nz:dz])
    msy,msz=np.meshgrid(idx_skipy,idx_skipz)
    upat[msy,msz] = 1
    idx_centy,idx_centz = np.int_(np.r_[max(0,ny//2-np.ceil(centy/2)+1):min(ny,ny//2+centy//2+1)]), \
                          np.int_(np.r_[max(0,nz//2-np.ceil(centz/2)+1):min(nz,nz//2+centz//2+1)])
    mcx,mcy = np.meshgrid(idx_centy,idx_centz)
    upat[mcx,mcy] = 1;
    return upat
    
def apply_pat(ksp_ncoil,pat_ny_nz):
    return pat_ny_nz[...,None,None]*ksp_ncoil

def do_coil_ffts(ksp):
    return bart(1, 'fft -iu 3', ksp)

def get_sens_maps(ksp):
    # 1 set of maps (m=1)
    return bart(1,'ecalib -m 1',ksp) 

def app_sens_maps(img2D,maps):
    denom = 1e-10+np.sum(maps*np.conj(maps),axis=3)
    return np.squeeze(np.sum(np.conj(maps)*img2D,axis=3)/denom)

def simple_sense_recon(ksp):
    # returns recon in img-space and sens maps given ksp
    sens = get_sens_maps(ksp)
    return bart(1,'ecalib -m 1',ksp) 
    LAMBDA=0.001
    ITER=25
    img_l2 = bart(1,'pics -S -l2 -r %f -i %d -d 4' % (LAMBDA,ITER),ksp,sens)
    return img_l2,sens

def complex_noise_pollute(ksp,snr_db):
    # return ksp with complex noise added to reach snr given in 20log10 dB
    s = np.percentile(np.abs(ksp.ravel()),99)
    sgma = s/np.power(10,snr_db/20)/np.sqrt(2)
    w = sgma*np.random.randn(*ksp.shape) + sgma*1j*np.random.randn(*ksp.shape)
    ksp_poll = ksp + w
    return ksp_poll

def nrmse(mat1,mat2):
    # root-mean-sq error between complex matrices, normalized by second arg rms as in bart
    return np.sqrt(np.sum(np.power(np.abs(mat1-mat2),2)))/np.sqrt(np.sum(np.power(np.abs(mat2),2)))

def est_snr_db(image_coils):
    return 20*np.log10(np.percentile(np.abs(image_coils),99)/(np.std(image_coils[:10,:10,...].real)/np.sqrt(2)))

def get_dataset(data_nm='bart_phantom',snr_db=15):
    if data_nm == 'bart_phantom':
        ksp_fully_sampled = bart(1, 'phantom -k -s 8 -x 256')
        ksp_fully_sampled = complex_noise_pollute(ksp_fully_sampled,snr_db)
        disp(np.abs(ksp_fully_sampled[...,0])**0.2,'ksp phantom ** 0.2',subplt=(1,3,1))
    elif data_nm == 'invivo_brain':
        ksp_fully_sampled = cfl.readcfl('kspace_brain').transpose((2,0,1))
        ksp_fully_sampled = ksp_fully_sampled[:,:,:,None].transpose([1,2,3,0])
        M,N,_,C=(np.shape(ksp_fully_sampled))
        # print('ksp dimensions as M,N,C:{}'.format([M,N,C]))
        disp(np.linalg.norm(bart(1,'fft -iu 3',ksp_fully_sampled),axis=3),'RSS fully-sampled brain image')
    elif data_nm[-3:] == '.h5':
        with h5py.File(data_nm, 'r') as F:
            # disp(F['reconstruction_rss'][0,...].squeeze())
            kspace = np.array(F['kspace'])
            ksp_fully_sampled = kspace[0,...].squeeze() # take first slice
            ksp_fully_sampled = ksp_fully_sampled[...,None].transpose([1,2,3,0])
            M,N,_,C=(np.shape(ksp_fully_sampled))
            image_space_coils = bart(1,'fft -iu 3',ksp_fully_sampled)
            image_space_coils = image_space_coils[int(M/4):int(3*M/4),...]
            image_space_coils = np.flipud(image_space_coils) # correct orientation
            ksp_fully_sampled = bart(1,'fft -u 3',image_space_coils)
    else:
        raise Exception('unknown dataset name!')
    return ksp_fully_sampled

def do_sense_recon( ksp_phantom_usamp, phantom_gt,lmbds=np.power(np.float32(10),np.r_[-5:0.5:0.5]),verbose=True ):
    maps = get_sens_maps(ksp_phantom_usamp) # 1 set of sensitivity maps
    nrmse_out = np.inf

    for il,LAMBDA in enumerate(lmbds):
        ITER=25
        phantom_l2 = bart(1,'pics -S -l2 -r %f -i %d -d 0' % (LAMBDA,ITER),ksp_phantom_usamp,maps)
        fksp_l2 = bart(1,'fakeksp -r',phantom_l2,ksp_phantom_usamp,maps)
        phantom_l2_fksp = app_sens_maps(bart(1,'fft -iu 3',fksp_l2),maps)
        
        nrmse_curr = nrmse(phantom_l2,phantom_gt)
        print('lambda = %f, NRMSE = %f' % (LAMBDA,np.round( nrmse_curr , 3 )))
        disp(phantom_l2,'l2 reg=1e{}'.format(int(np.log10(LAMBDA))),subplt=(2,len(lmbds),1+il))
        disp(phantom_l2-phantom_gt,plt_phase=False,subplt=(2,len(lmbds),1+len(lmbds)+il))
        if nrmse_curr < nrmse_out:
            nrmse_out = nrmse_curr
            phantom_l2_out = phantom_l2
            fksp_l2_out = fksp_l2



    return maps,phantom_l2_out,fksp_l2_out,nrmse_out

def gen_upat_2d(ny,nz,R,ncent,do_z_usamp=None):
    # returns 2d uniform sampling pattern of ny x nz, filled in center of centy x centz, else skipping w/ dy, dz
    assert ny%2==0 and nz%2==0, 'only even dimensions supported'
    do_z_usamp = nz<ny if do_z_usamp is None else do_z_usamp
    print('Doing z usamp: {}'.format(str(do_z_usamp)))
    upat = np.zeros([ny,nz])
    if do_z_usamp:
        dy,dz=1,R
        centy,centz=ny,ncent
    else:
        dy,dz=R,1  
        centy,centz=ncent,ny
    idx_skipy,idx_skipz = np.int_(np.r_[0:ny:dy]),np.int_(np.r_[0:nz:dz])
    msy,msz=np.meshgrid(idx_skipy,idx_skipz)
    upat[msy,msz] = 1
    idx_centy,idx_centz = np.int_(np.r_[max(0,ny//2-np.ceil(centy/2)+1):min(ny,ny//2+centy//2+1)]), \
                          np.int_(np.r_[max(0,nz//2-np.ceil(centz/2)+1):min(nz,nz//2+centz//2+1)])
    mcx,mcy = np.meshgrid(idx_centy,idx_centz)
    upat[mcx,mcy] = 1;
    return upat

def nrmse_retro_func(args):
    r_r,fksp_l2_pro,phantom_l2_fill = args
    M,N,_,C = fksp_l2_pro.shape
    ksp_upat_retro = apply_pat(fksp_l2_pro,gen_upat_2d(M,N,r_r,24))
    _,phantom_l2_acc,fksp_l2_retro,nrmse_retro = \
                    do_sense_recon( ksp_upat_retro,phantom_l2_fill ) 
    return (nrmse_retro,phantom_l2_acc)

def fn_param_sense_exper(data_nm):
    fn_out = 'sense_experiment_out/'+data_nm.split('/')[1].split('.')[0]
    if len(glob(fn_out+'_nrmse.npy'))>0:
        print('Found completed run for',fn_out)
        return
    return pro_retro_sense_experiment(fn_out,data_nm,\
        r_pro_vec = [1,2,4],r_retro_vec = np.r_[1:9:1])

def snr_param_phantom_exper(snr_db):
    t_int = int(time()*1e10%1e7)
    fn_out = 'phantom_experiment_out/' + 'snrdb{}_{}'.format(snr_db,t_int)
    np.random.seed(t_int)
    return pro_retro_sense_experiment(fn_out=fn_out, data_nm='bart_phantom', snr_db=snr_db)

def pro_retro_sense_experiment(fn_out=None, data_nm='bart_phantom', snr_db=30, \
                               r_pro_vec = [1,2,4],r_retro_vec = np.r_[1:9],\
                               n_bart_threads=1,verbose=True):
    os.environ['OMP_NUM_THREADS'] = str(n_bart_threads)
    do_z_usamp = True if data_nm[-3:]=='.h5' else None

    ksp_fully_sampled = get_dataset(data_nm,snr_db=snr_db)
    M,N,_,C = ksp_fully_sampled.shape
    n_pro = len(r_pro_vec); n_retro = len(r_retro_vec)
    
    nrmse_out = 100*np.ones([n_pro,n_retro])
    images_l2_out = np.zeros([n_pro,n_retro,M,N],dtype=np.complex64)
    for i,r_p in enumerate(r_pro_vec):
        
        maps = get_sens_maps(ksp_fully_sampled) # 1 set of sensitivity maps
        image_coils = bart(1, 'fft -iu 3', ksp_fully_sampled)
        est_snr_db = 20*np.log10(np.percentile(np.abs(image_coils),99)/np.mean(np.abs(image_coils[:10:10,0,0]))/np.sqrt(2))
        image_gt = app_sens_maps(image_coils,maps)

        start_pro_iter = time()
        ksp_upat_pro = apply_pat(ksp_fully_sampled, gen_upat_2d(M,N,r_p,24,do_z_usamp=do_z_usamp) )
        _,image_l2_pro,fksp_l2_pro,nrmse_pro = \
                            do_sense_recon( ksp_upat_pro,image_gt ) 
        if verbose:
            print('r_pro = %d, nrmse = %f' % (r_p, np.round(nrmse_pro,3)))

        for j,r_r in enumerate(r_retro_vec):
                
            ksp_upat_retro = apply_pat(fksp_l2_pro,gen_upat_2d(M,N,r_r,24,do_z_usamp=do_z_usamp))
            _,image_l2_retro,fksp_l2_retro,nrmse_retro = \
                            do_sense_recon( ksp_upat_retro,image_l2_pro,verbose=verbose ) 
            if verbose:
                print('r_retro = %d, nmrse = %f' % (r_r, np.round(nrmse_retro,3)))

            nrmse_out[i,j] = nrmse_retro
            images_l2_out[i,j,...] = image_l2_retro

        print('Completed subexperiment for r_pro = {} in {} sec'.format(r_p,time()-start_pro_iter))
    if fn_out != None:
        np.save(fn_out+'_nrmse',nrmse_out)
        np.save(fn_out+'_r_pro_vec',r_pro_vec,r_retro_vec)
        np.save(fn_out+'_r_retro_vec',r_retro_vec)
        np.save(fn_out+'_images_l2',images_l2_out)
    return nrmse_out,images_l2_out
