import GPy
import meshio
import numpy as np
points=meshio.read("data/bunny_0.ply").points
import matplotlib.pyplot as plt
from tqdm import trange
import pickle
input_dim=5

all_points=np.zeros((600,points.shape[0],points.shape[1]))
for i in trange(600):
    all_points[i]=meshio.read("data/bunny_"+str(i)+".ply").points



kernel = GPy.kern.RBF(input_dim, 1)+GPy.kern.Matern52(5,ARD=True) + GPy.kern.White(input_dim)

all_points=all_points.reshape(all_points.shape[0],-1)

Q = input_dim
m_gplvm = GPy.models.GPLVM(Y=all_points.reshape(all_points.shape[0],-1), input_dim=Q, kernel=kernel,normalizer=True)
m_gplvm.kern.lengthscale = .2
m_gplvm.kern.variance = 1
m_gplvm.likelihood.variance = 1.
print(m_gplvm)
m_gplvm.optimize(messages=1, max_iters=300)
latent=np.array(m_gplvm.X)
rec=m_gplvm.predict(latent)[0]
print(latent.shape)
print(m_gplvm.predict(latent)[0].shape)
np.save("latent.npy",latent)
print(np.linalg.norm(m_gplvm.predict(latent)[0]-all_points.reshape(all_points.shape[0],-1))/np.linalg.norm(all_points.reshape(all_points.shape[0],-1)))
print("Data var is",np.mean(np.var(all_points.reshape(all_points.shape[0],-1),axis=0)))
print("Recon var is",np.mean(np.var(rec.reshape(rec.shape[0],-1),axis=0)))
with open('decoder.pkl', 'wb') as file:
    pickle.dump(m_gplvm, file)