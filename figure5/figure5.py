import sys
sys.path.append("..")
from sense_experiment import *

fn_prefix = '../data/fastMRI/'
precomputed = False

### run experiment for synthetic data ###

if not precomputed:
	r_retro_vec = np.r_[1:9:1] 
	r_pro_vec = [1,2,4]
	data_nm = 'bart_phantom'
	n_bart_threads = 8
	verbose = True
	f = open(fn_prefix+'data/fastMRI/req_fastMRI.txt')
	fns = f.read().split('\n')
	f.close()
	for fn in fns[:300]:
		pro_retro_sense_experiment(fn_out=fn_prefix+'_{}'.format(fn), data_nm=fn, snr_db=snr_db, \
	                               r_retro_vec = r_retro_vec, r_pro_vec=r_pro_vec, \
	                                n_bart_threads=n_bart_threads,verbose=verbose)


### plot from saved data ###

plt.figure(figsize=(12,8))
for j_plt,fn in enumerate(fns):

	if j_plt==0:
		nrmse_mat = np.load(fn_prefix+'_snrdb{}'.format(snr_db)+'_nrmse.npy')/len(fns)
		r_retro_vec = np.load(fn_prefix+'_snrdb{}'.format(snr_db)+'_r_retro_vec.npy')
		r_pro_vec = np.load(fn_prefix+'_snrdb{}'.format(snr_db)+'_r_pro_vec.npy')
	else:
		nrmse_mat += np.load(fn_prefix+'_snrdb{}'.format(snr_db)+'_nrmse.npy')/len(fns)

ftsz=24
plt.plot(r_retro_vec,nrmse_mat,linewidth=4)
plt.rc('font', size=ftsz)
plt.xticks(r_retro_vec, size=ftsz); plt.yticks(np.r_[0:0.8:0.1],size=ftsz);
plt.legend(['$R_{pro}$' +'={}'.format(r_pro) if r_pro>1 else '$R_{pro}$' +'={}'.format(r_pro) for r_pro in r_pro_vec],fontsize=ftsz);
plt.xlabel('$R_{retro}$',size=ftsz)
plt.ylabel('NRMSE',size=ftsz);plt.ylim([0, 0.7]);
plt.title('SNR = {} dB'.format(snr_db));

plt.show()

