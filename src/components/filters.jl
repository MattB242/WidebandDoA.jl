
"""
    inter_sensor_delay(ϕ, Δx, c)

Compute the inter-sensor delay matrix \$D \\in \\mathbb{R}^{M \\times K}\$ in seconds for a linear array, where `M = length(isd)` is the number of sensors on the array, and `K = length(isd)` is the number of targets.
The matrix is computed as follows:
```math
{[D]}_{m,k} = \\frac{\\Delta x[m] \\, \\sin(\\phi[k])}{c}
```

# Arguments
* `ϕ::AbstractVector`: Vector of DoAs in radian. 
* `Δx::AbstractVector`: Vector of inter-sensor delay of the array.
* `c`: Propagation speed of the medium.

# Returns
* `delays`: Matrix containing the delays in seconds. Each row correspond to sensor, and each column correspond to the source.
"""
function inter_sensor_delay(ϕ::AbstractVector, Δx::AbstractVector, c)
    Tullio.@tullio threads=false τ[m,k] := Δx[m]*sin(ϕ[k])/c
end

"""
    WindowedSinc(n_fft) <: AbstractDelayFilter

Closed-form fractional delay filter by Pei and Lai[^PL2012][^PL2014]

# Arguments
* `n_fft::Int`: Number of taps of the filter.

[^PL2012]: S. -C. Pei and Y. -C. Lai, "Closed Form Variable Fractional Time Delay Using FFT," *IEEE Signal Processing Letters*, 2012.
[^PL2014]: S. -C. Pei and Y. -C. Lai, "Closed form variable fractional delay using FFT with transition band trade-off," In *Proceedings of the IEEE International Symposium on Circuits and Systems* (ISCAS), 2014.
"""
struct WindowedSinc <: AbstractDelayFilter
    n_fft::Int
end

function array_delay(filter::WindowedSinc, Δn::Matrix{T})  where {T<:Real}
    n_fft = filter.n_fft
    θ     = collect(0:n_fft-1)*2*T(π)/n_fft
    a_fd  = T(0.25)
    Tullio.@tullio H[n,m,k] := begin
        if (n - 1) <= floor(Int, n_fft/2)
            if (n - 1) == 0
                Complex{T}(1.0)
            elseif (n - 1) <= ceil(Int, n_fft/2) - 2
                exp(-1im*-Δn[m,k]*θ[n])
            elseif (n - 1) <= ceil(Int, n_fft/2) - 1
                a_fd*cos(-Δn[m,k]*T(π)) +
                    (1 - a_fd)*exp(-1im*-Δn[m,k]*2*T(π)/n_fft*(T(n_fft)/2 - 1))
            elseif (n - 1) == ceil(Int, n_fft/2)
                Complex{T}(cos(-Δn[m,k]*T(π)))
            end
        else
            zero(Complex{T})
        end
    end
    idx_begin_cplx = ceil(Int, n_fft/2) + 1
    H[idx_begin_cplx:end,:,:] = begin
        if isodd(n_fft)
            conj.(H[idx_begin_cplx-1:-1:2,:,:])
        else
            conj.(H[idx_begin_cplx:-1:2,:,:])
        end
    end
    H
end

function array_delay(filter::WindowedSinc, Δn::Matrix{T})  where {T<:Real}
    n_fft = filter.n_fft
    θ     = CuArray(collect(0:n_fft-1)*2*T(π)/n_fft)
    a_fd  = T(0.25)
    M, K  = size(Δn)

    Δn_gpu = CuArray(Δn)
    H      = CuArray(zeros(Complex{T}, n_fft, M, K))

    # Precompute cutoff indices to avoid host-only calls on GPU
    idx_c0 = 0
    idx_c1 = ceil(Int, n_fft/2) - 2
    idx_c2 = ceil(Int, n_fft/2) - 1
    idx_c3 = ceil(Int, n_fft/2)

    @tullio H[n,m,k] := begin
        idx = n - 1
        Δθ  = -Δn_gpu[m,k] * θ[n]

        idx == idx_c0 ? Complex{T}(1.0) :
        idx <= idx_c1 ? exp(im * Δθ) :
        idx == idx_c2 ? a_fd * cos(-Δn_gpu[m,k] * T(π)) +
                        (1 - a_fd) * exp(im * -Δn_gpu[m,k] * 2T(π)/n_fft * (T(n_fft)/2 - 1)) :
        idx == idx_c3 ? Complex{T}(cos(-Δn_gpu[m,k] * T(π))) :
        zero(Complex{T})
    end

    idx_start = ceil(Int, n_fft/2) + 1
    if isodd(n_fft)
        H[idx_start:end, :, :] .= conj.(H[idx_start-1:-1:2, :, :])
    else
        H[idx_start:end, :, :] .= conj.(H[idx_start:-1:2, :, :])
    end

    return H
end

struct ComplexShift <: AbstractDelayFilter
    n_fft::Int
end

function array_delay(filter::ComplexShift, Δn::Matrix{T}) where {T <: Real}
    n_fft = filter.n_fft
    ω     = collect(0:n_fft-1)*2*T(π)/n_fft
    Tullio.@tullio H[n,m,k] := exp(1im*Δn[m,k]*ω[n])
end

function array_delay(filter::ComplexShift, Δn::Matrix{T}) where {T <: Real}
    n_fft = filter.n_fft
    ω     = CuArray(collect(0:n_fft-1)*2*T(π)/n_fft)
    Δn_gpu = CuArray(Δn)
    Tullio.@tullio H[n,m,k] := exp(1im*Δn_gpu[m,k]*ω[n])
end