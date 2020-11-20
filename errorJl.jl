module errorJl


struct ErrorJl <: Exception
    message::String
end

Base.showerror(io::IO, e::ErrorJl) = print(io,"Error: $(e.message)")


end ## errorJl
