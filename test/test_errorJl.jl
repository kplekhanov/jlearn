using Test

include("errorJl.jl")
using .errorJl: ErrorJl


@test_throws ErrorJl throw(ErrorJl("Test error message"))
