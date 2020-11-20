module activFons


activation_list = ["id", "sigmoid", "relu", "leaky_relu"]

function check_activation(activation)
    return activation in activation_list
end

function id(z)     
    return z
end

function id_d(z)     
    return 1
end

function sigmoid(z)     
    return 1 / (1 + exp(-z))
end

function sigmoid_d(z)
    return z * (1 - z)
end

function relu(z)
    return Int(z > 0) * z
end

function relu_d(z)
    return Int(z > 0)
end

function leaky_relu(z, coef=0.01)
    return Int(z > 0) * z + Int(z < 0) * coef * z
end

function leaky_relu_d(z, coef=0.01)
    return Int(z > 0) - coef * Int(z < 0)
end


end ## activeFons
