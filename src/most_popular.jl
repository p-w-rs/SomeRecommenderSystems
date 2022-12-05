using Recommendation, SparseArrays
using DataFrames, CSV

links = DataFrame(CSV.File("../ml-small/links.csv"))
movies = DataFrame(CSV.File("../ml-small/movies.csv"))
ratings = DataFrame(CSV.File("../ml-small/ratings.csv"))
tags = DataFrame(CSV.File("../ml-small/tags.csv"))

userId_to_idx = zeros(Int, maximum(ratings[:,"userId"]))
for (i, id) in enumerate(unique(ratings[:,"userId"]))
    userId_to_idx[id] = i
end
movieId_to_idx = zeros(Int, maximum(movies[:,"movieId"]))
for (i, id) in enumerate(unique(movies[:,"movieId"]))
    movieId_to_idx[id] = i
end

function apply_map(uid, mid, r)
    return userId_to_idx[uid], movieId_to_idx[mid], r
end

n_users, n_items = length(unique(ratings[:,"userId"])), length(unique(movies[:,"movieId"]))
events = [Event(apply_map(ratings[i,1:3]...)...) for i in 1:nrow(ratings)]
data = DataAccessor(events, n_users, n_items)

recommender = MostPopular(data)
fit!(recommender)

userId = 1
top_k = 10
recs = recommend(recommender, userId, top_k, collect(1:n_items))
println(movies[first.(recs), :])