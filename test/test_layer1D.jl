using Test

include("layer1D.jl")
using .layer1D: Layer1D, forward, ErrorJl


@testset "Layer" begin
    @test_throws ErrorJl Layer1D(4, activation="lalala")
    layer = Layer1D(4)
    @test layer.n_features == 4
    @test layer.values == [0.0; 0.0; 0.0; 0.0]

    @test_throws ErrorJl Layer1D([1; 2; 4; 5], activation="lalala")
    layer = Layer1D([1; 2; 4; 5])
    @test layer.n_features == 4
    @test layer.values == [1.0; 2.0; 4.0; 5.0]

    layer = Layer1D(4)
    @test_throws ErrorJl forward(layer, [4.0; 5.0; 1.0; 2.0])

    layer = Layer1D(4, activation="id")
    forward(layer, [4.0; 5.0; -1.0; -2.0])
    @test layer.values == [4.0; 5.0; -1.0; -2.0]
end
