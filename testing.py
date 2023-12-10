import GPy
import meshio
import numpy as np
points=meshio.read("data/bunny_0.ply").points
import matplotlib.pyplot as plt
from tqdm import trange
import pickle




all_points=np.zeros((600,points.shape[0],points.shape[1]))
for i in trange(600):
    all_points[i]=meshio.read("data/bunny_test_"+str(i)+".ply").points

all_points=all_points.reshape(all_points.shape[0],-1)
input_dim=all_points.shape[1]
latent=np.load("latent.npy")



with open('decoder.pkl', 'rb') as file:
    m_dec=pickle.load(file)

new_lat=m_dec.infer_newX(all_points, optimize=False)
print(new_lat)
new_lat=new_lat[0]
rec=m_dec.predict(new_lat)[0]
print(np.linalg.norm(m_dec.predict(new_lat)[0]-all_points.reshape(all_points.shape[0],-1))/np.linalg.norm(all_points.reshape(all_points.shape[0],-1)))
print("Data var is",np.mean(np.var(all_points.reshape(all_points.shape[0],-1),axis=0)))
print("Recon var is",np.mean(np.var(rec.reshape(rec.shape[0],-1),axis=0)))
for i in trange(600):
    meshio.write_points_cells("data/bunny_rec_"+str(i)+"_rec.ply",rec[i].reshape(-1,3),{"triangle":meshio.read("data/bunny_test_"+str(i)+".ply").cells_dict['triangle']})