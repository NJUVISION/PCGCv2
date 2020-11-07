import os
import numpy as np 
import subprocess

def gpcc_encode(filedir, bin_dir, show=False):
  """Compress point cloud losslessly using MPEG G-PCCv6. 
  You can download and install TMC13 from 
  http://mpegx.int-evry.fr/software/MPEG/PCC/TM/mpeg-pcc-tmc13
  """

  subp=subprocess.Popen('./utils/tmc3'+ 
                        ' --mode=0' + 
                        ' --positionQuantizationScale=1' + 
                        ' --trisoup_node_size_log2=0' + 
                        ' --ctxOccupancyReductionFactor=3' + 
                        ' --neighbourAvailBoundaryLog2=8' + 
                        ' --intra_pred_max_node_size_log2=6' + 
                        ' --inferredDirectCodingMode=0' + 
                        ' --uncompressedDataPath='+filedir + 
                        ' --compressedStreamPath='+bin_dir, 
                        shell=True, stdout=subprocess.PIPE)
  c=subp.stdout.readline()
  while c:
    if show:
      print(c)
    c=subp.stdout.readline()
  
  return 

def gpcc_decode(bin_dir, rec_dir, show=False):
  subp=subprocess.Popen('./utils/tmc3'+ 
                        ' --mode=1'+ 
                        ' --compressedStreamPath='+bin_dir+ 
                        ' --reconstructedDataPath='+rec_dir, 
                        shell=True, stdout=subprocess.PIPE)
  c=subp.stdout.readline()
  while c:
    if show:
      print(c)      
    c=subp.stdout.readline()
  
  return


def avs_pcc_encode(filedir, bin_dir, show=False):
  """Compress point cloud losslessly using MPEG G-PCCv6. 
  You can download and install TMC13 from 
  http://mpegx.int-evry.fr/software/MPEG/PCC/TM/mpeg-pcc-tmc13
  """

  subp=subprocess.Popen('./utils/avs-pcc-encoder '+ 
                        ' -i '+filedir + 
                        ' -b '+bin_dir + 
                        ' -gqs=1 ' +
                        ' -gof=1', 
                        shell=True, stdout=subprocess.PIPE)
  c=subp.stdout.readline()
  while c:
    if show:
      print(c)
    c=subp.stdout.readline()
  
  return 

def avs_pcc_decode(bin_dir, rec_dir, show=False):
  subp=subprocess.Popen('./utils/avs-pcc-decoder '+  
                        ' -b '+bin_dir+ 
                        ' -r '+rec_dir, 
                        shell=True, stdout=subprocess.PIPE)
  c=subp.stdout.readline()
  while c:
    if show:
      print(c)      
    c=subp.stdout.readline()
  
  return



def load_ply_data(filename):
  '''
  load data from ply file.
  '''

  f = open(filename)
  #1.read all points
  points = []
  for line in f:
    #only x,y,z
    wordslist = line.split(' ')
    try:
      x, y, z = float(wordslist[0]),float(wordslist[1]),float(wordslist[2])
    except ValueError:
      continue
    points.append([x,y,z])
  points = np.array(points)
  points = points.astype(np.int32)#np.uint8
  # print(filename,'\n','length:',points.shape)
  f.close()

  return points

def write_ply_data(filename, points):
  '''
  write data to ply file.
  '''
  if os.path.exists(filename):
    os.system('rm '+filename)
  f = open(filename,'a+')
  #print('data.shape:',data.shape)
  f.writelines(['ply\n','format ascii 1.0\n'])
  f.write('element vertex '+str(points.shape[0])+'\n')
  f.writelines(['property float x\n','property float y\n','property float z\n'])
  f.write('end_header\n')
  for _, point in enumerate(points):
    f.writelines([str(point[0]), ' ', str(point[1]), ' ',str(point[2]), '\n'])
  f.close() 

  return