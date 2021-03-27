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
from utils_best import get_score, get_recall, get_score_all, get_recall_all, fun_opt

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
  latency = job.Latency.values
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
  y_test_true = job.loc[test_idx, 'Label'].values        ## binary groundtruth for testing tasks
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

  ## Prepare index for Wrangler
  cutoff91 = int(0.92 * latency.shape[0])
  alpha91 = lat_sort.tolist()[cutoff91]
  test91_idx = job.index[(job['Latency'] >= alpha) & (job['Latency'] < alpha91)].tolist()

  ###################################################
  ######## Start nontime series experiments #########
  ###################################################

  pred_opt, pred_ipw_opt, tm_opt, tm_ipw_opt = fun_opt(X_train, Y_train, X_test, Y_test, y_test_true, rs, delta)

  ## Oversampling to increase training size
  OS = job.iloc[test91_idx]
  X_OS, Y_OS = OS.to_numpy()[:,1:-1], OS.to_numpy()[:,0]
  Y_train_cat = np.asarray([0]*X_train.shape[0]+ [1]*len(test91_idx))
  X_train_cat = np.concatenate((X_train, X_OS))
  sm = SMOTE(0.9, random_state=rs, k_neighbors=1)
  X_train_os, Y_train_os = sm.fit_resample(X_train_cat, Y_train_cat)
  ## Wrangler
  start_time = time.time()
  r_wrangler = SVC(kernel='linear').fit(X_train_os, Y_train_os)
  p_wrangler = r_wrangler.predict(X_test)
  tm_wrangler = (time.time() - start_time)/Y_test.shape[0]*1000
  print("Wrangler time  : {} ms".format(tm_wrangler))
  print(X_train.shape, X_train_os.shape)


  ## Get percentile results dataframe
  np_time = np.asarray([tm_opt, tm_ipw_opt, tm_wrangler])
  df_time = pd.DataFrame(np_time.reshape(1,-1), columns=['gb_opt','gb_ipw_opt','wrangler'])
  df_time.to_csv('{}/res_nts/time/Job{}_time.csv'.format(out,jobid))
  df_time

  ## Get results
  avg = "weighted"

  score_opt = get_score_all(y_test_true, pred_opt, alpha, avg)
  score_ipw_opt = get_score_all(y_test_true, pred_ipw_opt, alpha, avg)
  score_wrangler = get_score(y_test_true, p_wrangler, alpha, avg)

  recall_opt = get_recall_all(Y_test, y_test_true, pred_opt, alpha, BI) 
  recall_ipw_opt = get_recall_all(Y_test, y_test_true, pred_ipw_opt, alpha, BI) 
  recall_wrangler = get_recall(Y_test, y_test_true, p_wrangler, alpha, BI)

  ## Get metrics score dataframe
  np_acc = np.concatenate([np.asarray(score_opt).reshape(1,-1),
                           np.asarray(score_ipw_opt).reshape(1,-1),
                           np.asarray(score_wrangler).reshape(1,-1)])
  df_acc = pd.DataFrame(np_acc, columns=['TPR','FPR'], 
                        index=['gb_opt','gb_ipw_opt','wrangler'])
  df_acc.to_csv('{}/res_nts/acc/Job{}_acc.csv'.format(out, jobid))
  print("Total TPR/FPR: ")
  print(df_acc)

  ## Get percentile results dataframe
  np_pct = np.concatenate([np.asarray(recall_opt).reshape(1,-1),
                           np.asarray(recall_ipw_opt).reshape(1,-1),
                           np.asarray(recall_wrangler).reshape(1,-1)])
  df_pct = pd.DataFrame(np_pct, columns=['<95','<99','99+'], 
                        index=['gb_opt', 'gb_ipw_opt', 'wrangler'])

  df_pct.to_csv('{}/res_nts/ptc/Job{}_pct.csv'.format(out,jobid))
  print("Tail percentile: ")
  print(df_pct)


if __name__ == '__main__':
  main()














