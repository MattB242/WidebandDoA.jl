using CUDA

function zero_upper!(A::CuArray)
    B, N, _ = size(A)
    total = B * N * N
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks zero_upper_kernel!(A, B, N)
end

@cuda function zero_upper_kernel!(A, B, N)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    total = B * N * N
    if idx <= total
        b = 1 + (idx - 1) % B
        ij = (idx - 1) ÷ B
        j = 1 + (ij ÷ N)
        i = 1 + (ij % N)
        if i < j
            A[b, i, j] = 0
        end
    end
    return A
end

function ldl_striped_matrix!(A::Array, ϵ = eps(real(eltype(A))))
    @assert size(A,2) == size(A,3)
    B = size(A,1)
    N = size(A,2)
    D = zeros(eltype(A), B, N)

    buf = zeros(eltype(A), B)

    @inbounds for j in 1:N
        Lj = view(A,1:B,j,1:j-1)
        Dj = view(D,1:B,1:j-1)
        @tullio buf[b] = abs2(Lj[b,k])*Dj[b,k]
        D[:,j] = A[:,j,j] - buf

        A[:,j,j] .= one(eltype(A))

        if any(@. abs(D[:,j]) ≤ ϵ)
            nothing, nothing
        end

        @inbounds for i in j+1:N
            Li = view(A,1:B,i,1:j-1)
            @tullio buf[b] = Li[b,k]*conj(Lj[b,k])*Dj[b,k]
            A[:,i,j] = (A[:,i,j] - buf) ./ D[:,j]
        end
    end
    @inbounds for j in 1:N, i in 1:j-1
        A[:,i,j] .= zero(eltype(A))
    end
    D, A
end

function ldl_striped_matrix!(A::CuArray{T,3}, ϵ = eps(real(T))) where T<:Union{ComplexF32, ComplexF64}
    
    @assert size(A,2) == size(A,3)
    B = size(A,1)
    N = size(A,2)
    D = CUDA.zeros(eltype(A), B, N)
    buf = CUDA.zeros(eltype(A), B)

    @allowscalar for j in 1:N
        if j > 1
            Lj = view(A, :, j, 1:j-1)
            Dj = view(D, :, 1:j-1)
            for b in 1:B
                acc = zero(eltype(A))
                for k in 1:j-1
                    acc += abs2(Lj[b,k])*Dj[b,k]
                end
                buf[b] = acc
            end
        else
            buf .= 0
        end
       
        D[:, j] .= A[:,j,j].-buf
        A[:,j,j] .= one(eltype(A))

        if any(abs.(D[:,j]) .≤ ϵ)
            return nothing, nothing  # fail-fast on singular block
        end

        for i in j+1:N
            Li = view(A,:, i,1:j-1)
            for b in 1:B
                acc = zero(eltype(A))
                for k in 1:j-1
                    acc += Li[b,k] * conj(Lk[b,k])*Dj[b,k]
                end
            A[b,i,j]=(A[b,i,j] - acc)/D[b,j]
            end
        end
    end

   
    zero_upper!(A)
    return D, A
end

function cholesky_striped_matrix!(A::Array, ϵ = eps(real(eltype(A))))
    @assert size(A,2) == size(A,3)
    B = size(A,1)
    N = size(A,2)
    buf = zeros(eltype(A), B)

    for j in 1:N
        Lj = view(A,1:B,j,1:j-1)
        @tullio buf[b] = abs2(Lj[b,k])
        A[:,j,j] = sqrt.(A[:,j,j] - buf)

        if any(@. abs(A[:,j,j]) ≤ ϵ)
            nothing
        end
        
        @inbounds for i in j+1:N
            Li = view(A,1:B,i,1:j-1)
            @tullio buf[b] = conj(Lj[k])*Li[k]
            A[:,i,j] = (A[:,i,j] - buf)./A[:,j,j]
        end
    end
    @inbounds for j in 1:N, i in 1:j-1
        A[:,i,j] .= zero(eltype(A))
    end
    A
end

function cholesky_striped_matrix!(A::CuArray{T,3}, ϵ = eps(real(T))) where T<:Union{ComplexF32, ComplexF64}
    @assert size(A,2) == size(A,3)
    B, N = size(A,1), size(A,2)
    buf = CUDA.zeros(T, B)

    @allowscalar for j in 1:N
        if j > 1
            Lj = view(A, :, j, 1:j-1)
            for b in 1:B
                acc = zero(T)
                for k in 1:j-1
                    acc += abs2(Lj[b,k])
                end
                buf[b] = acc
            end
        else
            buf .= 0
        end

        A[:,j,j] .= sqrt.(A[:,j,j] .- buf)

        if any(abs.(A[:,j,j]) .≤ ϵ)
            return nothing
        end

        for i in j+1:N
            Lj = view(A,:,j,1:j-1)
            Li = view(A,:,i,1:j-1)
            for b in 1:B
                acc = zero(T)
                for k in 1:j-1
                    acc += conj(Lj[b,k]) * Li[b,k]
                end
                A[b,i,j] = (A[b,i,j] - acc) / A[b,j,j]
            end
        end
    end

    zero_upper!(A)
    return A
end