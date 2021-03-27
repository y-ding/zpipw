import os
import numpy as np
import pandas as pd
import random
import time
import sys
import argparse

from os import listdir
from os.path import isfile, join

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor

from imblearn.over_sampling import SMOTE 

from utils_ts import fun_df_cumsum, fun_cum_vec, get_PCT, get_FPR, get_TPR, fun_cum_tpr

def main():

  #####################################################
  ########## Read data and simple processing ########## 
  #####################################################

  parser = argparse.ArgumentParser(description='Straggle Prediction on Live Data.')
  parser.add_argument('--data_path', type=str, help='Data path')
  parser.add_argument('--jobid', type=str, default="6343048076", help='Job ID') 
  parser.add_argument('--rs', type=int, default=42, help='Random state (default: 42)')
  parser.add_argument('--pt', type=float, default=0.04, help='Training set size (default: 0.2)')
  parser.add_argument('--tail', type=float, default=0.9, help='Latency threshold (default: 0.9)')
  parser.add_argument('--delta', type=float, default=0, help='Parameter for propensity score adjustment (default: 0)')
  parser.add_argument('--out', type=str, default='out', help='Output folder to save results (default: out)')

  args = parser.parse_args()
 
  path_ts   = args.data_path  
  jobid     = args.jobid
  delta     = args.delta  # Parameter to tune propensity score
  pt        = args.pt     # Training set size
  tail      = args.tail   # Latency threshold
  rs        = args.rs     # Random state
  out       = args.out    # Output folder to save results

  if not os.path.exists(out):
    sys.exit("No Result Folder Created!")

  print("data_path: {}".format(path_ts))
  print("jobid:     {}".format(jobid))
  print("delta:     {}".format(delta))
  print("pt   :     {}".format(pt))
  print("tail:      {}".format(tail))
  print("rs   :     {}".format(rs))
  print("out :      {}".format(out))

  path_ts_file = path_ts + jobid
  files_task = [f for f in listdir(path_ts_file) if isfile(join(path_ts_file, f))]
  job_rawsize = len(files_task)  ## Get number of tasks in a job

  task_colnames = ['ST','ET','JI', 'TI','MI', 'MCU', 'CMU', 'AMU', 'UPC', 'TPC', 'MAXMU', 'MIO',
            'MDK', 'MAXCPU', 'MAXIO', 'CPI', 'MAI', 'SP', 'AT', 'SCPU', 'EV', 'FL']
  task_fields = ['ST','ET','MCU', 'CMU', 'AMU', 'UPC', 'TPC', 'MAXMU', 'MIO',
            'MDK', 'MAXCPU', 'MAXIO', 'CPI', 'MAI', 'SCPU', 'EV', 'FL']
  task_cols = ['Latency','MCU', 'CMU', 'AMU', 'UPC', 'TPC', 'MAXMU', 'MIO',
            'MDK', 'MAXCPU', 'MAXIO', 'CPI', 'MAI', 'SCPU', 'EV', 'FL']

  ## Get cumulative time series data
  list_task = [] 
  list_tp = []  ## list of total period
  list_task_compact = []  ## list of last row
  for i in range(job_rawsize):
    task = pd.read_csv('{}/{}/{}'.format(path_ts,jobid,i), header=None,
                       names=task_colnames, usecols=task_fields)
    task_new, tp_new = fun_df_cumsum(task)
    list_tp.append(tp_new)
    list_task_compact.append(task_new.iloc[-1].tolist())
    list_task.append(task_new)

  ## Construct new non-time series data
  np_task_compact = np.array(list_task_compact)
  df_task_compact = pd.DataFrame(np_task_compact, columns=task_fields)
  df_task_compact['Latency'] = pd.Series(np.asarray(list_tp), index=df_task_compact.index)
  df_sel = df_task_compact[task_cols]
  job = (df_sel-df_sel.min())/(df_sel.max()-df_sel.min())
  job = job.dropna(axis='columns') 
  job_raw = job.reset_index(drop=True)

  ## Normalize task at different time points using final row
  list_task_nn = []
  ts_size = 0  ## max task size in a job
  cn_train = [i for i in list(job) if i not in ['Latency']]  
  for i in range(len(list_task)):    
    task = list_task[i][cn_train]
    task = (task-job[cn_train].min())/(job[cn_train].max()-job[cn_train].min())
    if ts_size < task.shape[0]:
        ts_size = task.shape[0]
    list_task_nn.append(task)

  #####################################################################################
  ########## Now we have complete job data constructed from time series data ########## 
  #####################################################################################

  ## Split training and testing data
  latency = job_raw.Latency.values
  ## Parameter to tune propensity score
  lat_sort = np.sort(latency)

  print("# tail :  {}".format(tail))
  print("# delta:  {}".format(delta))

  cutoff = int(tail*latency.shape[0])
  alpha = lat_sort.tolist()[cutoff]
  print("# alpha:  {}".format(alpha))

  cutoff_pt = int(pt * latency.shape[0])
  alpha_pt = lat_sort.tolist()[cutoff_pt]
  train_idx_init = job.index[job['Latency'] < alpha].tolist()
  test_idx_init = job.index[job['Latency'] >= alpha].tolist()
  train_idx_removed = job.index[(job['Latency'] >= alpha_pt) & (job['Latency'] < alpha)].tolist()
  print("# true tail: {}".format(len(test_idx_init)))

  train_idx = list(set(train_idx_init) - set(train_idx_removed))
  test_idx = test_idx_init + train_idx_removed
  print("# removed: {}".format(len(train_idx_removed)))

  job =job_raw.copy()  ## this is VERY IMPORTANT!!!
  job_train = job.iloc[train_idx]
  job_test = job.iloc[test_idx]
  print("# train: {}".format(job_train.shape[0]))
  print("# test:  {}".format(job_test.shape[0]))

  X_train = job_train.to_numpy()[:,1:]
  Y_train = job_train.to_numpy()[:,0]
  X_test = job_test.to_numpy()[:,1:]
  Y_test = job_test.to_numpy()[:,0]

  job.loc[train_idx_init, 'Label'] = 0
  job.loc[test_idx_init, 'Label'] = 1
  y_test_true = job.loc[test_idx, 'Label'].values ## binary groundtruth for testing tasks
  y_stra_true = job.loc[test_idx_init, 'Latency'].values ## groundtruth for straggler

  ## Get latency bins, [90,95), [95, 99), [99+]
  cutoff95 = int(0.95 * latency.shape[0])
  alpha95 = lat_sort.tolist()[cutoff95]
  cutoff99 = int(0.99 * latency.shape[0])
  alpha99 = lat_sort.tolist()[cutoff99]
  test95_idx = job.index[(job['Latency'] >= alpha) & (job['Latency'] < alpha95)].tolist()
  test99_idx = job.index[(job['Latency'] >= alpha95) & (job['Latency'] < alpha99)].tolist()
  test99p_idx = job.index[(job['Latency'] >= alpha99)].tolist()
  BI = np.cumsum([len(test95_idx), len(test99_idx), len(test99p_idx)])
  print("# latency bins: {}".format(BI))

  ###################################################
  ########## Start time series experiments ########## 
  ###################################################

  ## Padding zero rows to unify task size
  list_task_norm = []
  test_idx_gap = [i for i in test_idx if i not in test_idx_init]
  list_task_nn_stra = [list_task_nn[i] for i in test_idx_init]  ## only straggler tasks
  list_task_nn_gap = [list_task_nn[i] for i in test_idx_gap] ## nonstragglers in testing
  list_task_nn_test = [list_task_nn[i] for i in test_idx]  ## for all test tasks

  ss_stra, ss_gap = [d.shape[0] for d in list_task_nn_stra], [d.shape[0] for d in list_task_nn_gap]
  ts_init_size = np.max(ss_stra)   ## max task size/time intervals for stragglers
          
  for dd in list_task_nn:
    if dd.shape[0] < ts_init_size:
      df2 =  pd.DataFrame(np.zeros([(ts_init_size-dd.shape[0]),dd.shape[1]]), columns=list(dd))
      list_task_norm.append(dd.append(df2, ignore_index=True))
    else:
      list_task_norm.append(dd)       
                 
  ## Only care about tasks that are stragglers
  list_task_norm_stra = [list_task_norm[i] for i in test_idx_init]
  list_task_norm_gap = [list_task_norm[i] for i in test_idx_gap]
  list_task_norm_test = [list_task_norm[i] for i in test_idx]


  ###################################################
  ############ Wrangler training models ############# 
  ###################################################

  sm = SMOTE(0.3, random_state=rs, k_neighbors=1)
  pt23, pt16 = 2/3, 1/6

  train_id_23 = np.random.choice(train_idx_init, int(np.ceil(pt23*len(train_idx_init))), replace=False).tolist() + \
              np.random.choice(test_idx_init, int(np.ceil(pt23*len(test_idx_init))), replace=False).tolist()
  X_train_23, Y_train_23 = job.iloc[train_id_23].to_numpy()[:,1:-1], job.iloc[train_id_23].to_numpy()[:,-1]
  X_train_23_os, Y_train_23_os = sm.fit_resample(X_train_23, Y_train_23)   

  tm_wrangler_23_start = time.time()
  r_wrangler_23 = SVC(kernel='linear').fit(X_train_23_os, Y_train_23_os)
  tm_wrangler_23_end = time.time()
  tm_wrangler_23_train = tm_wrangler_23_end - tm_wrangler_23_start

  r_gb_23 = GradientBoostingClassifier(n_estimators=100, random_state=rs).fit(X_train_23_os, Y_train_23_os)

  train_id_16 = np.random.choice(train_idx_init, int(np.ceil(pt16*len(train_idx_init))), replace=False).tolist() + \
          np.random.choice(test_idx_init, int(np.ceil(pt16*len(test_idx_init))), replace=False).tolist()  
  X_train_16, Y_train_16 = job.iloc[train_id_16].to_numpy()[:,1:-1], job.iloc[train_id_16].to_numpy()[:,-1]
  X_train_16_os, Y_train_16_os = sm.fit_resample(X_train_16, Y_train_16)       

  tm_wrangler_16_start = time.time()
  r_wrangler_16 = SVC(kernel='linear').fit(X_train_16_os, Y_train_16_os)
  tm_wrangler_16_end = time.time()
  tm_wrangler_16_train = tm_wrangler_16_end - tm_wrangler_16_start

  r_gb_16 = GradientBoostingClassifier(n_estimators=100, random_state=rs).fit(X_train_16_os, Y_train_16_os)


  ###################################################
  ######Correlation and Sherlock training models#####
  ###################################################

  X_train_up, X_test_up, Y_train_up, Y_test_up = X_train, X_test, Y_train, Y_test
  ## Train base models
  r_gb = GradientBoostingRegressor(n_estimators=100, random_state=rs).fit(X_train_up, Y_train_up)

  lt_stra, lt_gap = len(list_task_norm_stra), len(list_task_norm_gap)  ## straggler/non-straggler size
  list_task_norm_gap_down = list_task_norm_gap

  kl_stra_gb,kl_stra_ipw_gb,kl_stra_wrangler_23,kl_stra_wrangler_16,kl_stra_gb_23, kl_stra_gb_16=[],[],[],[],[],[]

  kl_gap_gb,kl_gap_ipw_gb,kl_gap_wrangler_23,kl_gap_wrangler_16, kl_gap_gb_23, kl_gap_gb_16=[], [], [], [], [], []

  fl_gap_gb, fl_gap_ipw_gb, fl_gap_wrangler_23, fl_gap_wrangler_16,fl_gap_gb_23, fl_gap_gb_16=[], [], [], [], [], []

  tm_corr_list, tm_causal_list, tm_wrangler_23_list, tm_wrangler_16_list = [], [], [], []

  for k in range(2,ts_init_size):  # ts_init_size
    #print('kth row: {}'.format(k))
    
    p_stra_gb, p_stra_ipw_gb, p_stra_wrangler_23, p_stra_wrangler_16,p_stra_gb_23, p_stra_gb_16= \
    np.zeros(lt_stra),np.zeros(lt_stra),np.zeros(lt_stra),np.zeros(lt_stra),np.zeros(lt_stra),np.zeros(lt_stra)
        
    tn_stra = [i.iloc[k].values for i in list_task_norm_stra]
    np_tn_stra = np.asarray(tn_stra)
    np_tn_stra_nzidx = (np.where(np_tn_stra.any(axis=1))[0]).tolist()
    np_tn_stra_nz = np_tn_stra[~np.all(np_tn_stra == 0, axis=1)]
    
    tn_gap = [i.iloc[k].values for i in list_task_norm_gap_down]
    list_gap_idx = range(len(list_task_norm_gap_down))
    
    np_tn_gap = np.asarray(tn_gap)
    
    if len(np_tn_gap)>0:       
      #print('np_tn_gap:  {}'.format(np_tn_gap))
      np_tn_gap_zidx = (np.where(~np_tn_gap.any(axis=1))[0]).tolist()  ## indices of zero rows
      if len(np_tn_gap_zidx)>0:          
        tn_gap_pre = [list_task_norm_gap[i].iloc[k-1].values for i in np_tn_gap_zidx]
        np_tn_gap_pre = np.asarray(tn_gap_pre)
        
        p_gap_gb = r_gb.predict(np_tn_gap_pre).tolist()
        kl_gap_gb = kl_gap_gb + p_gap_gb 
        
        p_gap_wrangler_23 = r_wrangler_23.predict(np_tn_gap_pre).tolist()
        p_gap_wrangler_16 = r_wrangler_16.predict(np_tn_gap_pre).tolist()
        p_gap_gb_23 = r_gb_23.predict(np_tn_gap_pre).tolist()
        p_gap_gb_16 = r_gb_16.predict(np_tn_gap_pre).tolist()
        
        kl_gap_wrangler_23 = kl_gap_wrangler_23 + p_gap_wrangler_23 
        kl_gap_wrangler_16 = kl_gap_wrangler_16 + p_gap_wrangler_16
        kl_gap_gb_23 = kl_gap_gb_23 + p_gap_gb_23 
        kl_gap_gb_16 = kl_gap_gb_16 + p_gap_gb_16  
                    
        X = np.asarray(np.concatenate((X_train, np_tn_stra_nz, np_tn_gap_pre)))
        y = np.asarray([0] * X_train.shape[0] + [1] * np_tn_stra_nz.shape[0]+ [1] * np_tn_gap_pre.shape[0])
        clf = LogisticRegression(random_state=rs, solver='lbfgs').fit(X, y)
        ps = clf.predict_proba(X)
        ps1 = ps[(X_train.shape[0]+np_tn_stra_nz.shape[0]):,0] + delta
        
        p_gap_ipw_gb = [x/y for x, y in zip(p_gap_gb, ps1.tolist())]
        kl_gap_ipw_gb = kl_gap_ipw_gb + p_gap_ipw_gb
        
        fl_gap_gb.append(sum([1 for i in kl_gap_gb if i>alpha])/len(kl_gap_gb))
        fl_gap_ipw_gb.append(sum([1 for i in kl_gap_ipw_gb if i>alpha])/len(kl_gap_ipw_gb))
        fl_gap_wrangler_23.append(sum([1 for i in kl_gap_wrangler_23 if i>alpha])/len(kl_gap_wrangler_23))
        fl_gap_wrangler_16.append(sum([1 for i in kl_gap_wrangler_16 if i>alpha])/len(kl_gap_wrangler_16))
        fl_gap_gb_23.append(sum([1 for i in kl_gap_gb_23 if i>alpha])/len(kl_gap_gb_23))
        fl_gap_gb_16.append(sum([1 for i in kl_gap_gb_16 if i>alpha])/len(kl_gap_gb_16))
        
        X_train_up = np.concatenate((X_train_up, X_test_up[np_tn_gap_zidx]))
        Y_train_up = np.concatenate((Y_train_up, Y_test_up[np_tn_gap_zidx]))        
        list_gap_idx = [i for i in list_gap_idx if i not in np_tn_gap_zidx]        
        list_task_norm_gap_down = [list_task_norm_gap_down[i] for i in list_gap_idx]
    
        r_gb = GradientBoostingRegressor(n_estimators=100, random_state=rs).fit(X_train_up, Y_train_up)           
    
    start_time = time.time()
    r_gb = GradientBoostingRegressor(n_estimators=10, random_state=rs).fit(X_train_up, Y_train_up)
    p_stra_gb[np_tn_stra_nzidx] = r_gb.predict(np_tn_stra_nz) 
    tm = time.time() - start_time
    
    ps_start = time.time()
    X = np.asarray(np.concatenate((X_train, np_tn_stra_nz)))
    y = np.asarray([0] * X_train.shape[0] + [1] * np_tn_stra_nz.shape[0])
    clf = LogisticRegression(random_state=rs, solver='lbfgs').fit(X, y)
    ps = clf.predict_proba(X)
    ps1 = ps[X_train.shape[0]:,0] + delta
    tm_ps = time.time() - ps_start
    
    ## Prediction by IPW
    start_time_ipw = time.time()
    p_stra_ipw_gb[np_tn_stra_nzidx] = p_stra_gb[np_tn_stra_nzidx]/ ps1 
    tm_ipw = time.time() - start_time_ipw + tm_ps + tm
     
    tm_corr_list.append(tm/Y_test.shape[0]*1000)
    tm_causal_list.append(tm_ipw/Y_test.shape[0]*1000)
    
    tm_wrangler_23_pred_start = time.time()
    p_stra_wrangler_23[np_tn_stra_nzidx] = r_wrangler_23.predict(np_tn_stra_nz)
    tm_wrangler_23_pred_end = time.time()
    p_stra_wrangler_16[np_tn_stra_nzidx] = r_wrangler_16.predict(np_tn_stra_nz)
    tm_wrangler_16_pred_end = time.time()
    
    tm_wrangler_23 = tm_wrangler_23_pred_end-tm_wrangler_23_pred_start + tm_wrangler_16_train
    tm_wrangler_16 = tm_wrangler_16_pred_end-tm_wrangler_23_pred_end + tm_wrangler_23_train
    
    tm_wrangler_23_list.append(tm_wrangler_23/Y_test.shape[0]*1000)
    tm_wrangler_16_list.append(tm_wrangler_16/Y_test.shape[0]*1000)    
    
    p_stra_gb_23[np_tn_stra_nzidx] = r_gb_23.predict(np_tn_stra_nz)
    p_stra_gb_16[np_tn_stra_nzidx] = r_gb_16.predict(np_tn_stra_nz)
    
    kl_stra_gb.append(p_stra_gb)
    kl_stra_ipw_gb.append(p_stra_ipw_gb)
    
    kl_stra_wrangler_23.append(p_stra_wrangler_23)
    kl_stra_wrangler_16.append(p_stra_wrangler_16)
    kl_stra_gb_23.append(p_stra_gb_23)
    kl_stra_gb_16.append(p_stra_gb_16)    

  ## Get time results dataframe
  np_time = np.asarray([sum(tm_corr_list)/len(tm_corr_list),sum(tm_causal_list)/len(tm_causal_list),
                          sum(tm_wrangler_16_list)/len(tm_wrangler_16_list),
                          sum(tm_wrangler_23_list)/len(tm_wrangler_23_list)])
  df_time = pd.DataFrame(np_time.reshape(1,-1), columns=['gb', 'gb_ipw','wrangler_23', 'wrangler_16'])
  df_time.to_csv('{}/res_ts/time/Job{}_time.csv'.format(out,jobid))
  print("Time:")
  print(df_time)

  #### Get percentile tail results

  PCT_gb = get_PCT(kl_stra_gb, y_stra_true, alpha, BI)
  PCT_ipw_gb = get_PCT(kl_stra_ipw_gb, y_stra_true, alpha, BI)

  PCT_wrangler_23 = get_PCT(kl_stra_wrangler_23, y_stra_true, alpha, BI)
  PCT_wrangler_16 = get_PCT(kl_stra_wrangler_16, y_stra_true, alpha, BI)

  PCT_gb_23 = get_PCT(kl_stra_gb_23, y_stra_true, alpha, BI)
  PCT_gb_16 = get_PCT(kl_stra_gb_16, y_stra_true, alpha, BI)

  ## Get percentile results dataframe
  np_pct = np.concatenate([np.asarray(PCT_gb).reshape(1,-1),np.asarray(PCT_ipw_gb).reshape(1,-1),
                          np.asarray(PCT_wrangler_23).reshape(1,-1),np.asarray(PCT_wrangler_16).reshape(1,-1),
                          np.asarray(PCT_gb_23).reshape(1,-1),np.asarray(PCT_gb_16).reshape(1,-1)])
  df_pct = pd.DataFrame(np_pct, columns=['<95','<99','99+'], 
                        index=['gb', 'gb_ipw','wrangler_23', 'wrangler_16','gb_23', 'gb_16'])
  df_pct.to_csv('{}/res_ts/ptc/Job{}_pct.csv'.format(out,jobid))
  print("Tail percentile: ")
  print(df_pct)

  ## Get true positive rate
  TPR_gb = get_TPR(kl_stra_gb, alpha)
  TPR_ipw_gb = get_TPR(kl_stra_ipw_gb, alpha)
  TPR_wrangler_23 = get_TPR(kl_stra_wrangler_23, alpha)
  TPR_wrangler_16 = get_TPR(kl_stra_wrangler_16, alpha)
  TPR_gb_23 = get_TPR(kl_stra_gb_23, alpha)
  TPR_gb_16 = get_TPR(kl_stra_gb_16, alpha)

  ## Get false positive rate
  FPR_gb = get_FPR(kl_gap_gb, alpha)
  FPR_ipw_gb = get_FPR(kl_gap_ipw_gb, alpha)

  FPR_wrangler_23 = get_FPR(kl_gap_wrangler_23, alpha)
  FPR_wrangler_16 = get_FPR(kl_gap_wrangler_16, alpha)
  FPR_gb_23 = get_FPR(kl_gap_gb_23, alpha)
  FPR_gb_16 = get_FPR(kl_gap_gb_16, alpha)

  TPR_L = [TPR_gb, TPR_ipw_gb, TPR_wrangler_23, TPR_wrangler_16, TPR_gb_23, TPR_gb_16]
  FPR_L = [FPR_gb, FPR_ipw_gb, FPR_wrangler_23, FPR_wrangler_16, FPR_gb_23, FPR_gb_16]
  df_acc = pd.DataFrame(list(zip(TPR_L,FPR_L)), columns=['TPR', 'FPR'],
                       index=['gb', 'gb_ipw','wrangler_23', 'wrangler_16','gb_23', 'gb_16'])
  df_acc.to_csv('{}/res_ts/acc/Job{}_acc.csv'.format(out, jobid))
  print("Total TPR/FPR: ")
  print(df_acc)

  ## Get TPR CDF
  pr_tpr_gb = fun_cum_tpr(kl_stra_gb, alpha)
  pr_tpr_ipw_gb = fun_cum_tpr(kl_stra_ipw_gb, alpha)

  pr_tpr_wrangler_23 = fun_cum_tpr(kl_stra_wrangler_23, alpha)
  pr_tpr_wrangler_16 = fun_cum_tpr(kl_stra_wrangler_16, alpha)
  pr_tpr_gb_23 = fun_cum_tpr(kl_stra_gb_23, alpha)
  pr_tpr_gb_16 = fun_cum_tpr(kl_stra_gb_16, alpha)

  pr_tpr = pr_tpr_gb
  pr_tpr_ipw = pr_tpr_ipw_gb

  df_cdf_tpr = pd.DataFrame(list(zip(pr_tpr, pr_tpr_ipw, pr_tpr_wrangler_23, pr_tpr_wrangler_16, pr_tpr_gb_23, pr_tpr_gb_16)), 
                            columns=['Correlation', 'Causal', 'Wrangler_23', 'Wrangler_16', 'GB_23', 'GB_16'])
  df_cdf_tpr.to_csv('{}/res_ts/cdf_tpr/Job{}_cdf_tpr.csv'.format(out, jobid))
  print("TPR CDF: ")
  print(df_cdf_tpr)

  ## Get FPR CDF
  pr_fpr = [x * 100 for x in fl_gap_gb]  
  pr_fpr_ipw = [x * 100 for x in fl_gap_ipw_gb] 

  pr_fpr_wrangler_23 = [x * 100 for x in fl_gap_wrangler_23] 
  pr_fpr_wrangler_16 = [x * 100 for x in fl_gap_wrangler_16]

  pr_fpr_gb_23 = [x * 100 for x in fl_gap_gb_23] 
  pr_fpr_gb_16 = [x * 100 for x in fl_gap_gb_16]

  df_cdf_fpr = pd.DataFrame(list(zip(pr_fpr,pr_fpr_ipw, pr_fpr_wrangler_23,pr_fpr_wrangler_16, pr_fpr_gb_23, pr_fpr_gb_16)), 
                            columns=['Correlation', 'Causal', 'Wrangler_23', 'Wrangler_16', 'GB_23', 'GB_16'])

  #df_cdf_fpr.to_csv('{}/res_ts/cdf_fpr/Job{}_cdf_fpr.csv'.format(out, jobid))
  print("FPR CDF: ")
  print(df_cdf_fpr)

if __name__ == '__main__':
  main()














