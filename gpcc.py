import os
import numpy as np 
import subprocess
rootdir = os.path.split(__file__)[0]

def gpcc_encode(filedir, bin_dir, show=False):
    """Compress point cloud losslessly using MPEG G-PCCv12. 
    You can download and install TMC13 from 
    https://github.com/MPEGGroup/mpeg-pcc-tmc13
    """
    subp=subprocess.Popen(rootdir+'/tmc3'+ 
                            ' --mode=0' + 
                            ' --positionQuantizationScale=1' + 
                            ' --trisoupNodeSizeLog2=0' + 
                            ' --neighbourAvailBoundaryLog2=8' + 
                            ' --intra_pred_max_node_size_log2=6' + 
                            ' --inferredDirectCodingMode=0' + 
                            ' --maxNumQtBtBeforeOt=4' +
                            ' --uncompressedDataPath='+filedir + 
                            ' --compressedStreamPath='+bin_dir, 
                            shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: print(c)
        c=subp.stdout.readline()
    
    return 

def gpcc_decode(bin_dir, rec_dir, show=False):
    subp=subprocess.Popen(rootdir+'/tmc3'+ 
                            ' --mode=1'+ 
                            ' --compressedStreamPath='+bin_dir+ 
                            ' --reconstructedDataPath='+rec_dir+
                            ' --outputBinaryPly=0'
                          ,
                            shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: print(c)      
        c=subp.stdout.readline()
    
    return