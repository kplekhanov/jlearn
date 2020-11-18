module errorJl


struct ErrorJl <: Exception
    message::String
end

Base.showerror(io::IO, e::ErrorJl) = print(io, e.message)


end ## errorJl
