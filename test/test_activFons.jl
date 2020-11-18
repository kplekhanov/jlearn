using Test

include("activFons.jl")
using .activFons


@testset "Activation functions" begin
    for activation in ["sigmoid", "relu", "leaky_relu", "None"]
        @test activFons.check_activation(activation) == true
    end
    @test activFons.check_activation("lalala") == false

    @test activFons.id(0.123) == 0.123
    
    @test activFons.sigmoid(0.0) == 0.5
    @test activFons.sigmoid_d(0.0) == 0.0

    @test activFons.relu(0.0) == 0.0
    @test activFons.relu(-10.0) == 0.0
    @test activFons.relu(10.0) == 10.0

    @test activFons.relu_d(0.0) == 0.0
    @test activFons.relu_d(-10.0) == 0.0
    @test activFons.relu_d(10.0) == 1.0

    @test activFons.leaky_relu(0.0) == 0.0
    @test activFons.leaky_relu(-10.0) == -0.1
    @test activFons.leaky_relu(10.0) == 10.0

    @test activFons.leaky_relu_d(0.0) == 0.0
    @test activFons.leaky_relu_d(-10.0) == -0.01
    @test activFons.leaky_relu_d(10.0) == 1.0
end
