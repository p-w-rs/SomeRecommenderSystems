import datatable as dt
import torch as pt
import numpy as np

from tqdm import tqdm

# Get cpu or gpu device for training.
# device = "cuda" if pt.cuda.is_available() else "cpu"
# print(f"Using {device} device")
device = "cpu"

genome_scores = dt.fread("./ml-25m/genome-scores.csv")
genome_tags = dt.fread("./ml-25m/genome-tags.csv")
links = dt.fread("./ml-25m/links.csv")
movies = dt.fread("./ml-25m/movies.csv")
ratings = dt.fread("./ml-25m/ratings.csv")
tags = dt.fread("./ml-25m/tags.csv")

nm = movies.shape[0]
mxid = dt.max(movies["movieId"])
mxid = 209171 + 1
mid_midx = pt.zeros(mxid, dtype=pt.int64)
midx_mid = pt.zeros(mxid, dtype=pt.int64)
for i in tqdm(range(nm)):
    mid_midx[movies[i, "movieId"]] = i
    midx_mid[i] = movies[i, "movieId"]

pt.save(mid_midx, "SAVES/ItemBasedCF_gpu/mid_midx.pt")
pt.save(midx_mid, "SAVES/ItemBasedCF_gpu/midx_mid.pt")

nu = dt.unique(ratings["userId"]).shape[0]
print("size:", (nu, nm))
UxM = pt.zeros((nu, nm))
for k in tqdm(range(ratings.shape[0])):
    midx = mid_midx[ratings[k, "movieId"]]
    uidx = ratings[k, "userId"] - 1
    UxM[uidx, midx] = ratings[k, "rating"]

pt.save(UxM, "SAVES/ItemBasedCF_gpu/UxM.pt")
# MxU = MxU.to(device)

h = pt.tensor(5)  # .to(device)
nM = pt.linalg.norm(UxM, dim=2).reshape(nm, 1)
nMxM = nM @ nM.T
pt.save(nMxM.to("cpu"), "SAVES/ItemBasedCF_gpu/nMxM.pt")

similarity = (UxM.T @ UxM) * pt.reciprocal(nMxM.add(h))
pt.save(similarity, "SAVES/ItemBasedCF_gpu/similarity.pt")
