from simplex_PE import *

def opt_wrapper(args):
    
    num_run = args[-2]
    r_idx = args[-1]
    init_pars = args[:-1]
    
#     sys.stdout.write("Initial parameters {0}: {1}\n".format(int(r_idx), init_pars))
    
    SAVE_DIR = './Simulated_Grid/ODE/Cidx_01_Lidx_10/run_1/opt_pars/'
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    
    SAVE_PATH = SAVE_DIR+'R'+str(int(r_idx))+'.npy'
    if not os.path.isfile(SAVE_PATH):
        #par_dicts = opt_par_search_rand(true_df,estim_params,fixed_params,frame_vec,crocker_metric,args[:-1])
        par_dicts = opt_par_search(true_df,estim_params,fixed_params,frame_vec,crocker_metric)
        save_dict = {}
        save_dict['par_dicts'] = par_dicts
        save_dict['init_pars'] = init_pars
        np.save(SAVE_PATH,save_dict)