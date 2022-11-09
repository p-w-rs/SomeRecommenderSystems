using SparseArrays, c, DataFrames, CSV, JLD2, ProgressMeter

genome_scores = DataFrame(CSV.File("./ml-25m/genome-scores.csv"))
genome_tags = DataFrame(CSV.File("./ml-25m/genome-tags.csv"))
links = DataFrame(CSV.File("./ml-25m/links.csv"))
movies = DataFrame(CSV.File("./ml-25m/movies.csv"))
ratings = DataFrame(CSV.File("./ml-25m/ratings.csv"))
tags = DataFrame(CSV.File("./ml-25m/tags.csv"))


mxid = maximum(movies[!, "movieId"])
mid_midx = zeros(Int32, mxid)
midx_mid = zeros(Int32, mxid)
@showprogress for i in 1:nrow(movies)
    mid_midx[movies[i, "movieId"]] = i
    midx_mid[i] = movies[i, "movieId"]
end

jldopen("SAVES/ItemBasedCF.jld2", "w") do f
    f["mid_midx"] = mid_midx
    f["midx_mid"] = midx_mid
end

nm = nrow(movies)
nu = length(unique(ratings[!, "userId"]))
println("size:", (nm, nu))
MxU = zeros(Float32, nm, nu)
@showprogress for k in 1:nrow(ratings)
    midx = mid_midx[ratings[k, "movieId"]]
    uidx = ratings[k, "userId"]
    MxU[midx, uidx] = ratings[k, "rating"]
end

jldopen("SAVES/ItemBasedCF.jld2", "a+") do f
    f["MxU"] = MxU
end

R = zeros(Float32, nm, nm)
pairwise!(R, CosineDist(), MxU, dims=1)