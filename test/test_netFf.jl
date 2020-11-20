using Test

include("../netFf.jl")
using .netFf


@testset "Feed-forward NN" begin
    net = netFf.NetFf()
    netFf.add!(net, netFf.layer1D.Layer1D(4; activation="id"))
    netFf.add!(net, netFf.layer1D.Layer1D(8; activation="relu"))
    netFf.add!(net, netFf.layer1D.Layer1D(1; activation="sigmoid"))
    netFf.initialize!(net)

    netFf.set_x!(net, [0 1 2 3; 0 1 2 3; 0 1 2 3; 0 1 2 3])
    netFf.set_y!(net, [4 4 4 4])
    
    netFf.train!(net; n_iter=100)
end
