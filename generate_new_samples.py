import torch
import pickle
import meshio
from tqdm import trange
with open('decoder.pkl', 'rb') as file:
    m_dec=pickle.load(file)


model=torch.load("nf.pt")
model.eval()


for i in trange(600):
    new_sample=model.sample(1)[0]
    new_sample=new_sample.detach().numpy()
    rec=m_dec.predict(new_sample)[0].reshape(-1,3)
    meshio.write_points_cells("data/bunny_rec_nf_"+str(i)+".ply",rec.reshape(-1,3),{"triangle":meshio.read("data/bunny_"+str(i)+".ply").cells_dict['triangle']})
