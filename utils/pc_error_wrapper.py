import numpy as np 
import os, time
import pandas as pd
import subprocess

def get_points_number(filedir):
    plyfile = open(filedir)

    line = plyfile.readline()
    while line.find("element vertex") == -1:
        line = plyfile.readline()
    number = int(line.split(' ')[-1][:-1])
    
    return number

def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try:
            number = float(item) 
        except ValueError:
            continue
        
    return number

def pc_error(infile1, infile2, res, normal=False, show=False):
    # Symmetric Metrics. D1 mse, D1 hausdorff.
    headers1 = ["mse1      (p2point)", "mse1,PSNR (p2point)", 
               "h.       1(p2point)", "h.,PSNR  1(p2point)" ]

    headers2 = ["mse2      (p2point)", "mse2,PSNR (p2point)", 
               "h.       2(p2point)", "h.,PSNR  2(p2point)" ]

    headersF = ["mseF      (p2point)", "mseF,PSNR (p2point)", 
               "h.        (p2point)", "h.,PSNR   (p2point)" ]

    haders_p2plane = ["mse1      (p2plane)", "mse1,PSNR (p2plane)",
                      "mse2      (p2plane)", "mse2,PSNR (p2plane)",
                      "mseF      (p2plane)", "mseF,PSNR (p2plane)"]

    headers = headers1 + headers2 + headersF

    command = str('./utils/pc_error_d' + 
                          ' -a '+infile1+ 
                          ' -b '+infile2+ 
                        #   ' -n '+infile1+
                          ' --hausdorff=1 '+ 
                          ' --resolution='+str(res-1))

    if normal:
      headers += haders_p2plane
      command = str(command + ' -n ' + infile1)

    results = {}
   
    start = time.time()
    subp=subprocess.Popen(command, 
                          shell=True, stdout=subprocess.PIPE)

    c=subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')# python3.
        if show:
            print(line)
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value

        c=subp.stdout.readline() 
    print('===== measure PCC quality using `pc_error` version 0.13.4', round(time.time() - start, 4))

    return pd.DataFrame([results])

def avs_pcc_pc_evalue(infile1, infile2, res, show=False):
    # Evaluate using AVS PCC PC evaluation tool.
    # D1 mse, D1 hausdorff.
    
    # 1. Take original point cloud as reference:
    headers_1 = ["D1_MSE_1", "D1_PSNR_1", 
               "D1_Hausdorff_1", "D1_HausdorffPSNR_1"]
    # 2. Take reconstruct point cloud as reference:
    headers_2 = ["D1_MSE_2", "D1_PSNR_2", 
               "D1_Hausdorff_2", "D1_HausdorffPSNR_2"]
    # 3. Symmetric result:
    headers_f = ["D1_MSE_F", "D1_PSNR_F", 
               "D1_Hausdorff_F", "D1_HausdorffPSNR_F"]
    headers = headers_1 + headers_2 + headers_f
    results = {}
    
    start = time.time()
    subp=subprocess.Popen('./utils/avs-pcc-pc_evalue '+ 
                          ' -f1 '+infile1+ 
                          ' -f2 '+infile2+ 
                          ' -pk '+str(res-1), 
                          shell=True, stdout=subprocess.PIPE)

    c=subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')# python3.
        if show:
            print(line)

        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value

        c=subp.stdout.readline() 
    print('===== measure PCC quality using `avs-pcc-pc_evalue` v0.1', round(time.time() - start, 4))

    return pd.DataFrame([results])
