module netFf

using StaticArrays

include("layer1D.jl")
using .layer1D

include("errorJl.jl")
using .errorJl: ErrorJl


"""
x is the input; y is the output. Both are expected to be matrices
where cols -- samples and rows -- features (yeah, unconventional)
"""
mutable struct NetFf
    x::AbstractMatrix
    y::AbstractMatrix
	layers::Array{layer1D.Layer1D,1}
end

"""
creating a net with non-initialized entries and empty layer list
"""
function NetFf(x::AbstractMatrix=Array{Float64}(undef, 0, 0),
               y::AbstractMatrix=Array{Float64}(undef, 0, 0))
    return NetFf(x, y, [])
end

function set_x!(net::NetFf, x::AbstractMatrix)
    net.x = x
end

function set_y!(net::NetFf, y::AbstractMatrix)
    net.y = y
end

function add!(net::NetFf, layer::layer1D.Layer1D)
    push!(net.layers, layer)
    return net
end

function initialize!(net::NetFf)
    n_layers = length(net.layers)
    if n_layers < 2
        throw(ErrorJl("Can't initialize less than 2 layers of the NetFf"))
    end
    for i in 1:n_layers-1
        layer1D.initialize_lmap!(net.layers[i], net.layers[i+1].n_features)
    end
    net.layers[end].terminal = true
end

function forwardprop_sweep!(net::NetFf)
    h = net.x
    for layer in net.layers
        h = layer1D.forwardprop!(layer, h)
    end
end

function backprop_sweep!(net::NetFf; reg=0.0, lrate=0.01)
    delta = net.layers[end].values - net.y
    for layer in reverse(net.layers)
        delta = layer1D.backprop!(layer, delta, reg, lrate)
    end
end

function train!(net::NetFf; n_iter=100, reg=0.0, lrate=0.01)
    for i_iter in 1:n_iter
        println(i_iter)
        forwardprop_sweep!(net)
        backprop_sweep!(net; reg=reg, lrate=lrate)
    end
end

end ## netFf
