import open3d as o3d
import numpy as np
import random
import os, time
from data_utils import write_ply_ascii_geo, write_h5_geo

def sample_points(mesh_filedir, n_points=4e5, resolution=255):
    # sample points uniformly.
    mesh = o3d.io.read_triangle_mesh(mesh_filedir)
    try:
        pcd = mesh.sample_points_uniformly(number_of_points=int(n_points))
    except:
        print("ERROR sample_points", '!'*8)
        return
    points = np.asarray(pcd.points)
    return points

def get_rotate_matrix():
    m = np.eye(3,dtype='float32')
    m[0,0] *= np.random.randint(0,2)*2-1
    m = np.dot(m,np.linalg.qr(np.random.randn(3,3))[0])

    return m

def mesh2pc(mesh_filedir, n_points, resolution):
    points = sample_points(mesh_filedir, n_points=n_points, resolution=resolution)
    # random rotate.
    points = np.dot(points, get_rotate_matrix())
    # normalize to fixed resolution.
    points = points - np.min(points)
    points = points / np.max(points)
    points = points * (resolution)
    # quantizate to integers.
    points = np.round(points).astype('int')
    points = np.unique(points, axis=0)

    return points

def generate_dataset(mesh_filedirs, pc_rootdir, out_filetype, n_points=4e5, resolution=255):
    start_time = time.time()
    for idx, mesh_filedir in enumerate(mesh_filedirs):
        try: 
            points = mesh2pc(mesh_filedir, n_points, resolution)
        except:
            print("ERROR generate_dataset", idx, '!'*8)
            continue
        if out_filetype == 'ply':
            pc_filedir = os.path.join(pc_rootdir, str(idx) + '_' \
                                        + os.path.split(mesh_filedir)[-1].split('.')[0] + '.ply')
            write_ply_ascii_geo(pc_filedir, points)
        if out_filetype == 'h5':
            pc_filedir = os.path.join(pc_rootdir, str(idx) + '_' \
                                        + os.path.split(mesh_filedir)[-1].split('.')[0] + '.h5')
            write_h5_geo(pc_filedir, points)
        if idx % 100 == 0: print('='*20, idx, round((time.time() - start_time)/60.), 'mins', '='*20)

    return 

def traverse_path_recursively(rootdir):
    filedirs = []
    def gci(filepath):
        files = os.listdir(filepath)
        for fi in files:
            fi_d = os.path.join(filepath,fi)            
            if os.path.isdir(fi_d):
                gci(fi_d)                  
            else:
                filedirs.append(os.path.join(filepath,fi_d))
        return
    gci(rootdir)

    return filedirs

if __name__ == "__main__":
    mesh_rootdir = "/home/ubuntu/HardDisk1/ModelNet40/"
    pc_rootdir = './dataset/'
    out_filetype = 'ply'
    out_filetype = 'h5'
    num_mesh = 100
    n_points = int(4e5) # dense
    resolution = 127

    input_filedirs = traverse_path_recursively(rootdir=mesh_rootdir)
    mesh_filedirs = [f for f in input_filedirs if (os.path.splitext(f)[1]=='.off' or os.path.splitext(f)[1]=='.obj')]# .off or .obj
    mesh_filedirs = random.sample(mesh_filedirs, num_mesh)
    print("mesh_filedirs:\n", len(input_filedirs), len(mesh_filedirs))
    if not os.path.exists(pc_rootdir): os.makedirs(pc_rootdir)

    generate_dataset(mesh_filedirs, pc_rootdir, out_filetype, n_points, resolution)

    


