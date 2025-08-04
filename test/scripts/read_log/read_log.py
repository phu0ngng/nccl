#!/usr/bin/env python3
import argparse
import re
import os
import numpy as np
import scipy.stats as stats
# change numpy display
np.set_printoptions(precision=4,threshold=4)


verbose = 0

# apply the Grubb's test to reject outliers with a confidence of (1-alpha)
# use the one-sided max-based filtering.
# retuns a filtered array of data, and the number of outliers found
def grubb_test_max(data,alpha,mask=None,nOut = 0):
    if(mask is None):
        if(verbose):
            print(f"Grubb's test on {data} with {100-100*alpha}% confidence")
        mask = np.ones(np.shape(data),dtype=int)
    x = data[np.where(mask > 0)]
    n = len(x)
    mean_x = np.mean(x)
    sd_x = np.std(x)
    max_x = np.max(x)
    numerator = max_x-mean_x
    g_calculated = 0.0
    if(sd_x > 0.0):
        g_calculated = numerator/sd_x
    # two sided: factor = alpha/(2*n)
    factor = alpha/(n)
    t_value_1 = stats.t.ppf(1 - factor, n - 2)
    g_critical = ((n - 1) * np.sqrt(np.square(t_value_1))) / (np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value_1)))
    # test if the max is an outlier
    if(g_calculated > g_critical and nOut < 10):
        idx = np.where(data == max_x)[0][0]
        assert(data[idx] == max_x)
        mask[idx] = 0
        if(verbose):
            print(f"\t> {max_x:.4f} @ data[{idx}] is an outlier (mean {mean_x:.4f}, test {g_calculated:.4f} > {g_critical:.4f})")
        return grubb_test_max(data,alpha,mask=mask,nOut=nOut+1)
    else:
        if(verbose):
            print(f"\t> {max_x:.4f} is no outlier (mean {mean_x:.4f}, test {g_calculated:.4f} <= {g_critical:.4f})")
        return nOut, mask



def readLog(fileName, pattern):
    idx = 0
    log = []
    with open(fileName, 'r') as file:
        if(verbose):
            print(f"looking in {fileName} for match with {pattern}")
        content = file.readlines()
        for line in content:
            # if the pattern is present, save it
            if pattern in line:
                dataline = line.split(pattern,1)
                log.insert(idx,dataline[1])
        file.close()
    return log


def myPrint(a):
    print(a.expandtabs(tabsize=8))
def toLen(a,size=10):
    return a.ljust(size)

# extract numbers with format ("float" or "int") from the lines matching pattern in the log file fileName
def extractFromLog(fileName,pattern,labels=None,nProcs=0, n_repeat=1,n_warmup=0,format="float",alpha = 0.05):
    N_DATA = 15
    file_to_open = f"{fileName}"
    # Open the file in read mode
    # try:
    content = readLog(file_to_open,pattern)
    # flter
    n_cols = 0
    n_data = len(content)
    result = np.zeros([n_data,N_DATA])
    c=0
    for line in content:
        if(format == "float"):
            data = re.findall("\\d+\\.\\d+",line)
        if(format == "int"):
            data = re.findall("\\d+",line)
        j=0
        for d in data:
            result[c,j] = float(d)
            j=j+1
        if(n_cols == 0):
            n_cols=j
        c=c+1
    # get the number of lines per repetition and remove the warmups
    if nProcs ==0:
        nProcs = int(n_data/n_repeat)
        if (n_data%n_repeat):
            print(f"WARNING: wrong number of data ({n_data}) or iterations ({n_repeat}), considering {n_repeat} iterations with {nProcs} processes")
    else:
        n_repeat = int(n_data/nProcs)
        if (n_data%nProcs):
            print(f"WARNING: wrong number of data ({n_data}) or processes ({nProcs}), considering {n_repeat} iterations with {nProcs} processes")
    nIter = n_repeat-n_warmup
    # results - per iterations
    lres_avg = np.zeros([nIter,n_cols])
    lres_min = np.zeros([nIter,n_cols])
    lres_max = np.zeros([nIter,n_cols])
    # the rolling sum^2 is used to get the CI over all the data
    nData = 0
    roll_sum = np.zeros([n_cols])
    roll_sum2 = np.zeros([n_cols])
    for i in range(int(n_warmup),int(n_repeat)):
        idx_min = int(i*nProcs)
        idx_max = int((i+1)*nProcs)
        n_idx = idx_max - idx_min
        local_res = result[idx_min:idx_max,0:n_cols]
        # get the average on the current iteration
        idx = i-n_warmup
        lres_avg[idx,:] = np.mean(local_res,axis=0)
        lres_min[idx,:] = np.min(local_res,axis=0)
        lres_max[idx,:] = np.max(local_res,axis=0)
        roll_sum[:] = roll_sum[:] + np.sum(local_res,axis=0)
        roll_sum2[:] = roll_sum2[:] + np.sum(np.square(local_res),axis=0)
        nData = nData + n_idx;
    # return the max and min in the error format
    avg = np.zeros([n_cols])
    min = np.zeros([n_cols])
    max = np.zeros([n_cols])
    ci = np.zeros([n_cols])
    std = np.zeros([n_cols])
    favg = np.zeros([n_cols])
    fmin = np.zeros([n_cols])
    fmax = np.zeros([n_cols])
    nOut = np.zeros([n_cols])
    for i in range(n_cols):
        # un-filtered statistics
        avg[i] = np.mean(lres_avg[:,i])
        max[i] = np.max(lres_max[:,i])
        min[i] = np.min(lres_min[:,i])
        # get the confidence interval
        #num = (nData * roll_sum2[i] - roll_sum[i]**2)
        sigma2 = roll_sum2[i]/nData - (roll_sum[i]/nData)**2
        if (sigma2 > 0.0 and nData>2):
            # sample variance
            std[i] = np.sqrt(sigma2)
            #unbiased sample variance
            s = np.sqrt( nData / (nData - 1) * sigma2);
            # t-student, look for the prob to be outside of the CI with prob alpha.
            # because of symmetry, need to look for the ppf with alpha/2
            t_factor = stats.t.ppf(1 - alpha/2.0, nData - 1)
            ci[i] = t_factor * s/np.sqrt(nData)
        else:
            std[i] = 0
            ci[i] = 0
        # remove the outliers and reobtain the avg on the filtered data
        nOut[i],mask = grubb_test_max(lres_avg[:,i],alpha)
        mask_idx = np.where(mask>0)
        favg[i] = np.mean(lres_avg[mask_idx,i])
        fmin[i] = np.min(lres_min[mask_idx,i])
        fmax[i] = np.max(lres_max[mask_idx,i])

    myPrint(f"======================================================================================================================================================================")
    myPrint(f"Results for \"{pattern}\" in {fileName}")
    myPrint(f"----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    header = ""
    myPrint(f"{toLen("")}\t{toLen("")}\t{toLen("")}\t{toLen(">> un-filtered <<",size=28)}\t{toLen("")}\t|\t{toLen("")}\t{toLen(">> filtered <<",size=28)}")
    myPrint(f"{toLen("",size=14)}--------------------------------------------------------------------------------------------------------------------------------------------------------")
    myPrint(f"{toLen(header)}\t{toLen("avg")}\t{toLen("std")}\t{toLen(f"CI {100-100*alpha}%")}\t{toLen("min")}\t{toLen("max")}\t|\t{toLen(f"out iters")}\t{toLen("f-avg")}\t{toLen(f"f-min")}\t{toLen(f"f-max")}")
    for i in range(n_cols):
        header = f"column #{i}"
        if labels is not None and i < len(labels):
            header = labels[i]
        myPrint(f"{toLen(header)}\t{toLen(f" {avg[i]:f}")}\t{toLen(f" {std[i]:f}")}\t {toLen(f" {ci[i]:f}")}\t {toLen(f"{min[i]:.2f}")}\t {toLen(f"{max[i]:.2f}")}\t|\t {toLen(f"{int(nOut[i])}/{nIter}")}\t {toLen(f"{favg[i]:f}")}\t {toLen(f"{fmin[i]}")}\t {toLen(f"{fmax[i]}")}")

    myPrint(f"----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    myPrint(f"legend:")
    myPrint(f"  {toLen("avg")}\taverage over the ranks and the iterations")
    myPrint(f"  {toLen("CI")}\twidth of the {100-100*alpha}% confidence interval (difference between the measured avg and the true avg with {100-100*alpha}% confidence)")
    myPrint(f"  {toLen("min")}\tmininum over the ranks and the iterations")
    myPrint(f"  {toLen("max")}\tmaximum over the ranks and the iterations")
    myPrint(f"  {toLen("out iters")}\tnumber of iterations detected as outlier with a confidence of {100-100*alpha}%")
    myPrint(f"  {toLen("f-avg")}\taverage over the filtered iterations")
    myPrint(f"  {toLen("f-min")}\tmin over the filtered iterations")
    myPrint(f"  {toLen("f-max")}\tmin over the filtered iterations")
    myPrint(f"Notes:")
    myPrint(f" - measurements using {n_repeat} iterations, removing the first {n_warmup} ones")
    myPrint(f" - filtered iterations are obtained by first averaging the ranks over a single iteations, and then remove the outlier iterations with confidence {100-100*alpha}% based on that average time")
    myPrint(f"======================================================================================================================================================================")

    # except:
    #     print(f"error when reading {fileName}")


parser = argparse.ArgumentParser()
parser.add_argument("fileName", help="the name of the NCCL log file to read")
parser.add_argument("pattern", help="the pattern to search for in the log file")
parser.add_argument("-v", "--verbose", help="increase output verbosity",action="store_true")
parser.add_argument("-p", "--procs",type=int,help="number of processes; if given, used to determine the number of iterations; if not, we will use the number of data / number of iterations", default=0)
parser.add_argument("-n", "--iters",type=int,help="number of iterations", default=1)
parser.add_argument("-w", "--warmup",type=int,help="number of warmup iterations (substracted from iterations)", default=0)
parser.add_argument("-a", "--alpha",type=float,help="confidence intervals and outlier percent is (1-alpha)*100", default=0.01)
parser.add_argument("-t", "--type",help="family of headers to be used, accepted values: init, alloc", default="init")
parser.add_argument("-l", "--labels",nargs='+', help="list of header for each for the measures", default=None)
args = parser.parse_args()

# set the verbosity
verbose = args.verbose
# read the data
labels = None
if (args.labels is not None):
    labels = args.labels
else:
    if(args.type == "init"):
        labels=["total","kernels","alloc","bootstrap","allgather","topo","graph","connect","rest"]
    elif (args.type == "alloc"):
        labels=["total","plugin","netinit","nvml","rest"]
    else:
        print(f"un-recognized type {args.type}, ignoring")

extractFromLog(args.fileName,args.pattern,labels=labels,n_repeat=args.iters,n_warmup=args.warmup,alpha=args.alpha,nProcs=args.procs)


# end of file
