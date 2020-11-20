using Test

include("../errorJl.jl")
using .errorJl: ErrorJl

@testset "Error" begin
    @test_throws ErrorJl throw(ErrorJl("Test error message"))
end
