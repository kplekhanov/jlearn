module layer1D

using StaticArrays

include("activFons.jl")
using .activFons

include("errorJl.jl")
using .errorJl: ErrorJl


"""
values is a matrix with cols -- samples and rows -- features
forward pass is splitted into an activation and a linear map (lmap)
lmap has as rows next (right) layer features and as cols -- actual layer features
"""
mutable struct Layer1D
    n_features::Int
    values::AbstractMatrix
    lmap::AbstractMatrix
    activation::String
    terminal::Bool
end

"""
check array and if it is a vector transforms it into a matrix
"""
function checked_mat(mat::AbstractArray)
    if length(size(mat)) == 1
        return reshape(mat, length(mat), 1)
    else
        return mat
    end
end

"""
creating a layer with non-initialized values and lmap
"""
function Layer1D(n_features::Int; activation="id", terminal=false)
    if !activFons.check_activation(activation)
        throw(ErrorJl("Activation function not in the list"))
    end
    values = Array{Float64}(undef, 0, 0)
    lmap = Array{Float64}(undef, 0, 0)
    return Layer1D(n_features, values, lmap, activation, terminal)
end

"""
initialize values from an input matrix into a static internal matrix
"""
function initialize_values!(layer::Layer1D, mat::AbstractArray)
    if layer.n_features != size(mat)[1]
        throw(ErrorJl("Shapes of the layer and the values matrix do not match"))
    end
    mat = checked_mat(mat)
    n_rows, n_cols = size(mat)
    layer.values = SMatrix{n_rows,n_cols}(mat)
end

"""
initialize values with zero from an int into a static internal matrix
"""
function initialize_values!(layer::Layer1D, n_cols::Int)
    layer.values = @SMatrix zeros(layer.n_features, n_cols)
end

"""
initialize linear map from a matrix
"""
function initialize_lmap!(layer::Layer1D, mat::AbstractMatrix)
    n_rows, n_cols = size(mat)
    if layer.n_features != n_cols
        throw(ErrorJl("Shapes of the layer and the lmap matrix do not match"))
    end
    layer.lmap = SMatrix{n_rows,n_cols}(mat)
end

"""
initialize linear map with random values from an int
"""
function initialize_lmap!(layer::Layer1D, n_rows::Int)
    layer.lmap = @SMatrix rand(n_rows, layer.n_features)
end

"""
forward pass through the activation and the linear map
the terminal layer does not need an lmap so it just returns values
"""
function forwardprop!(layer::Layer1D, mat::AbstractArray)
    if layer.n_features != size(mat)[1]
        throw(ErrorJl("Shapes of the forwardprop data and the lmap matrix do not match"))
    end
    activation_fon = getfield(activFons, Symbol(layer.activation))
    layer.values = activation_fon.(checked_mat(mat))
    if layer.terminal == true    
        return layer.values
    else
        return layer.lmap * layer.values
    end
end

"""
backprop pass through the layer
mat is delta times the old activation derivative
the output is a new delta times the new activation derivative
the terminal layer just propagates delta to avoid 1 / (1 - z) division
"""
function backprop!(layer::Layer1D, mat::AbstractArray, reg, lrate)
    if layer.terminal == true
        return checked_mat(mat)
    elseif size(layer.lmap)[1] != size(mat)[1]
        throw(ErrorJl("Shapes of the backprop data and the lmap matrix do not match"))
    else
        d_activation_fon = getfield(activFons, Symbol(layer.activation * "_d"))
        resu = d_activation_fon.(layer.values) .* (transpose(layer.lmap) * checked_mat(mat))
        delta_lmap = checked_mat(mat) * transpose(layer.values) / size(layer.values)[2]
        layer.lmap -= (delta_lmap + reg * layer.lmap) * lrate
        return resu
    end
end


end ## layer1D
