module layer1D

using StaticArrays

include("activFons.jl")
using .activFons

include("errorJl.jl")
using .errorJl: ErrorJl


mutable struct Layer1D{S}
	n_features::Int
    values::SVector{S,Float64}
    activation::String
end

"""
creating a layer with zero values everywhere
"""
function Layer1D(n_features::Int; activation="None")
    if !activFons.check_activation(activation)
        throw(ErrorJl("Activation function not in the list"))
    end
    values = @SVector zeros(Float64, n_features)
    return Layer1D{n_features}(n_features, values, activation)
end

"""
creating a layer from a vector
"""
function Layer1D(values::Vector; activation="None")
    if !activFons.check_activation(activation)
        throw(ErrorJl("Activation function not in the list"))
    end
    n_features = length(values)
    return Layer1D{n_features}(n_features, values, activation)
end

"""
forward pass through the activation function
"""
function forward(layer::Layer1D, values_in::Vector)
    if layer.activation == "None"
        throw(ErrorJl("Can't forward prop the layer with no activation"))
    else
        activation_fon = getfield(activFons, Symbol(layer.activation))
        layer.values = activation_fon.(values_in)
    end
end


end ## layer1D
