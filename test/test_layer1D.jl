using Test

include("../layer1D.jl")
using .layer1D: ErrorJl, Layer1D, initialize_lmap!, initialize_values!,
    forwardprop!, backprop!


@testset "Layer" begin
    @test_throws ErrorJl Layer1D(3, activation="lalala")
    
    layer = Layer1D(3)
    @test layer.n_features == 3
    @test layer.activation == "id"

    ## testing values initialization
    initialize_values!(layer, 1)
    @test layer.values == [0.0; 0.0; 0.0][:,:]
    initialize_values!(layer, [0.0 0.0; 0.0 0.0; 0.0 0.0])
    @test layer.values == [0.0 0.0; 0.0 0.0; 0.0 0.0]
    @test_throws ErrorJl initialize_values!(layer, [0.0; 0.0])

    ## testing lmap initialization
    @test initialize_lmap!(layer, 10) != 0
    initialize_lmap!(layer, [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0])
    @test layer.lmap == [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0]
    @test_throws ErrorJl initialize_lmap!(layer, [1.0 0.0; 0.0 2.0; 3.0 0.0])
    
    ## testing forwardprop
    @test forwardprop!(layer, [1.0 2.0; 3.0 4.0; 5.0 6.0]) == [1.0 2.0; 6.0 8.0; 15.0 18.0]
    @test layer.values == [1.0 2.0; 3.0 4.0; 5.0 6.0]

    layer = Layer1D(3, activation="relu")
    initialize_lmap!(layer, [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0])
    @test forwardprop!(layer, [1.0; 2.0; -3.0]) == [1.0; 4.0; 0.0][:,:]
    @test layer.values == [1.0; 2.0; 0.0][:,:]
    
    layer = Layer1D(3, activation="leaky_relu")
    initialize_lmap!(layer, [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0])
    @test forwardprop!(layer, [1.0; 2.0; -3.0]) == [1.0; 4.0; -0.09][:,:]
    @test layer.values == [1.0; 2.0; -0.03][:,:]

    layer = Layer1D(3, terminal=true)
    initialize_lmap!(layer, [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0])
    @test forwardprop!(layer, [1.0; 2.0; 3.0]) == [1.0; 2.0; 3.0][:,:]
    @test layer.values == [1.0; 2.0; 3.0][:,:]

    ## testing backprop
    layer = Layer1D(3, activation="id")
    initialize_lmap!(layer, [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0])
    forwardprop!(layer, [1.0 2.0; 3.0 4.0; 5.0 6.0])
    @test backprop!(layer, [1.0 2.0; 3.0 4.0; 5.0 6.0]) == [1.0 2.0; 6.0 8.0; 15.0 18.0]

    layer = Layer1D(3, activation="id", terminal=true)
    @test backprop!(layer, [1.0; 2.0; 3.0]) == [1.0; 2.0; 3.0][:,:]
end
