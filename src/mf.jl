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
for (i, id) in enumerate(unique(ratings[:,"movieId"]))
    movieId_to_idx[id] = i
end

function apply_map(uid, mid, r)
    return userId_to_idx[uid], movieId_to_idx[mid], r
end

n_users, n_items = length(unique(ratings[:,"userId"])), length(unique(ratings[:,"movieId"]))
events = [Event(apply_map(ratings[i,1:3]...)...) for i in 1:nrow(ratings)]
data = DataAccessor(events, n_users, n_items)

n_factors = 2
recommender = FactorizationMachines(data, n_factors)
fit!(recommender, learning_rate=15e-4, max_iter=10)

userId = 1
favs = []
user_ratings = filter(row -> row["userId"] == userId, ratings)
for rating in eachrow(user_ratings)
    push!(favs, Pair(rating["rating"], movieId_to_idx[rating["movieId"]]))
end
println(movies[last.(sort(favs, rev=true)), :])

top_k = 20
recs = recommend(recommender, userId, top_k, collect(1:n_items))
println(movies[first.(recs), :])