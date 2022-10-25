import datatable as dt
import numpy as np

from tqdm import tqdm

genome_scores = dt.fread("./ml-25m/genome-scores.csv")
genome_tags = dt.fread("./ml-25m/genome-tags.csv")
links = dt.fread("./ml-25m/links.csv")
movies = dt.fread("./ml-25m/movies.csv")
ratings = dt.fread("./ml-25m/ratings.csv")
tags = dt.fread("./ml-25m/tags.csv")

nm = movies.shape[0]
mxid = dt.max(movies["movieId"])
mxid = 209171 + 1
mid_midx = np.zeros(mxid, np.int64)
midx_mid = np.zeros(mxid, np.int64)
for i in tqdm(range(nm)):
    mid_midx[movies[i, "movieId"]] = i
    midx_mid[i] = movies[i, "movieId"]

np.save("SAVES/ItemBasedCF_cpu/mid_midx.npy", mid_midx)
np.save("SAVES/ItemBasedCF_cpu/midx_mid.npy", midx_mid)

nu = dt.unique(ratings["userId"]).shape[0]
print("size:", (nm, nu))
MxU = np.zeros((nm, nu))
for k in tqdm(range(ratings.shape[0])):
    midx = mid_midx[ratings[k, "movieId"]]
    uidx = ratings[k, "userId"] - 1
    MxU[midx, uidx] = ratings[k, "rating"]

np.save("SAVES/ItemBasedCF_cpu/MxU.npy", MxU)

h = 5
nM = np.linalg.norm(MxU, axis=1, keepdims=True)
nMxM = np.matmul(nM, np.transpose(nM))
np.save("SAVES/ItemBasedCF_cpu/nMxM.npy", nMxM)

similarity = np.matmul(MxU, np.transpose(MxU)) * np.reciprocal(nMxM + h)
np.save("SAVES/ItemBasedCF_cpu/similarity.npy", similarity)

