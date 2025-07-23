using CUDA

"""
    WidebandIsoIsoLikelihood(n_samples, n_fft, delay_filter, Δx, c, fs)

Collapsed likelihood for a isotropic normal source prior and an isotropic normal noise prior.

# Arguments
* `n_samples::Int`: Number of samples in received signal.
* `n_fft::Int`: Length of the latent source signal.
* `delay_filter::AbstractDelayFilter`: Delay filter.
* `Δx::AbstractVector`: Inter-sensor delay in seconds.
* `c::Real`: Propagation speed of the medium in m/s.
* `fs::Real`: Sampling rate in Hz.

Given a parameter `NamedTuple(phi[j], loglambda[j])`, this likelihood computes:

```math
\\begin{aligned}
    &\\log p\\left(y \\mid \\phi_{1:k}, \\gamma_{1:k}, \\alpha, \\beta \\right) \\\\
    &= -\\frac{N + \\beta}{2} \\log\\left(\\frac{\\alpha}{2} + y^{\\dagger} {(H \\Lambda H^{\\dagger} + \\mathrm{I})}^{-1} y \\right)
       - \\frac{1}{2} \\det \\left(H^{\\dagger} \\Lambda H + \\mathrm{I}\\right)  \\\\
    &= -\\frac{N + \\beta}{2} \\log\\left(\\frac{\\alpha}{2}
       + y^{\\dagger} y - y^{\\dagger} H {\\left( \\Lambda^{-1} + H^{\\dagger} H \\right)}^{-1} H^{\\dagger} y \\right)
       - \\frac{1}{2} \\det\\left(\\Lambda\\right) \\det \\left(\\Lambda^{-1} + H^{\\dagger} H \\right), \\\\
\\end{aligned}
```
where
```math
\\Lambda = \\mathrm{diag}\\left(
    \\exp\\left( \\text{\\textsf{loglambda[0]}} \\right),
    \\ldots,
    \\exp\\left( \\text{\\textsf{loglambda[k]}} \\right)
 \\right)
```
(Note that \$\\gamma_j\$ in the paper is `lambda[j]` in the code, which is a bit confusing.)
"""
struct WidebandIsoIsoLikelihood{
    DF <: AbstractDelayFilter,
    AO <: AbstractVector,
    F  <: Real
} <: AbstractWidebandLikelihood
    n_samples   ::Int
    n_fft       ::Int
    delay_filter::DF
    Δx          ::AO
    c           ::F
    fs          ::F
end

function loglikelihood(
    likelihood::WidebandIsoIsoLikelihood,
    prior     ::WidebandIsoSourcePrior,
    data      ::WidebandData,
    params,
)
    @unpack y_fft, y_power = data
    @unpack n_fft, n_samples, delay_filter, Δx, c, fs = likelihood
    @unpack alpha, beta = prior

    ϕ = [param.phi            for param in params]
    λ = [exp(param.loglambda) for param in params]

    N = n_samples
    M = size(y_fft, 2)
    K = length(ϕ)

    if K == 0
        -(N*M/2 + beta)*log(alpha/2 + y_power)
    else
        τ  = inter_sensor_delay(ϕ, Δx, c)
        Δn = τ*fs
        H  = CuArray(array_delay_gpu(delay_filter, Δn))
        y_fft_gpu = CuArray(y_fft)

        HᴴH = CUDA.zeros(ComplexF64, size(H,1), size(H,3), size(H,3)) # '
        @tullio HᴴH[n,j,k] := conj(H[n,m,j]) * H[n,m,k]

        Λ⁻¹pHᴴH = CUDA.zeros(ComplexF64, size(H,1), size(H,3), size(H,3))
        @tullio Λ⁻¹pHᴴH[n,j,k] := HᴴH[n,j,k] + ((j == k) ? 1/λ[k] : 0)

        D, L = ldl_striped_matrix_gpu!(Λ⁻¹pHᴴH)

        ℓdetΛ⁻¹pHᴴH_gpu = CUDA.zeros(Float64, size(D,1))
        @tullio ℓdetΛ⁻¹pHᴴH_gpu := log(real(D[n,m]))
        if isnothing(L) || !all(isfinite, (ℓdetΛ⁻¹pHᴴH_gpu))
            return -Inf
        end

       
        ℓdetΛ = n_fft*sum(log, λ)
        ℓdetP⊥ = ℓdetΛ + sum(ℓdetΛ⁻¹pHᴴH_gpu)

        @tullio Hᴴy[n,k] := conj(H[n,m,k]) * y_fft_gpu[n,m]

        L⁻¹Hᴴy            = trsv_striped_matrix_gpu!(L, Hᴴy)

        yᴴImP⊥y_gpu = CUDA.zeros(Float74, suze(H,1))
        @tullio yᴴImP⊥y_gpu[n] := real(L⁻¹Hᴴy[n,i]/D[n,i]*conj(L⁻¹Hᴴy[n,i]))
        y_power_gpu = CuArray(fill(y_power, N))
        yᴴP⊥y_gpu            = y_power_gpu - yᴴImP⊥y_gpu

        if any(x -> x <= eps(Float64), Array(yᴴImP⊥y_gpu))
            return -Inf
        end

        yᴴP⊥y_total = sum(Array(yᴴP⊥y_gpu))
        ℓdetP⊥_total = ℓdetP⊥
        -(N*M/2 + beta)*log(alpha/2 + yᴴP⊥y_total) - ℓdetP⊥_total/2
    end
end


"""
    rand(rng, likelihood::WidebandIsoIsoLikelihood, x, phi; prior, sigma)

Sample from the collapsed likelihood for the model with isotropic normal prior and isotropic normal noise.

# Arguments
* `rng::Random.AbstractRNG`: Random number generator.
* `likelihood::WidebandIsoIsoLikelihood`: Likelihood.
* `x::AbstractMatrix`: Latent source signals, where rows are the signals and columns are sources.
* `phi::AbstractVector`: Direction-of-arrivals. 

# Keyword Arguments
* `prior`: Prior object used to sample `sigma` if needed (default: `nothing`).
* `sigma`: Signal standard deviation. (Default samples from `InverseGamma(prior.alpha, prior.beta)`)

# Returns
* `y`: A simulated received signal, where the rows are the channels (sensors) and the columns are received signals.

The sampling process is as follows:
```math
\\begin{aligned}
    \\epsilon &\\sim \\mathcal{N}(0, \\sigma^2 \\mathrm{I}) \\\\
    x         &\\sim \\mathcal{N}(0, \\sigma^2 H \\Lambda H^{\\top}) \\\\
    y         &= x + \\epsilon  
\\end{aligned}
```
and the noise \$\\epsilon\$,
```math
\\begin{aligned}
    y         &\\sim \\mathcal{N}(0, \\sigma^2 \\left( H \\Lambda H^{\\top} + \\mathrm{I} \\right)).
\\end{aligned}
```
Sampling from this distribution is as simple as
```math
\\begin{aligned}
  y = \\sigma H \\Lambda^{1/2} z_x + \\sigma z_{\\epsilon},
\\end{aligned}
```
where \$z_x\$ and \$z_{\\epsilon}\$ are independent standard Gaussian vectors.
"""
function Base.rand(
    rng       ::Random.AbstractRNG,
    likelihood::WidebandIsoIsoLikelihood,
    x         ::AbstractMatrix,
    phi       ::AbstractVector;
    prior     ::Union{<:AbstractWidebandPrior,Nothing} = nothing,
    sigma     ::Real = rand(rng, InverseGamma(prior.alpha, prior.beta)),
)
    @unpack n_samples, delay_filter, Δx, c, fs = likelihood

    N, M = n_samples, length(Δx)
    ϕ, σ = phi, sigma

    z_ϵ = randn(rng, N, M)

    k = length(ϕ)
    τ = inter_sensor_delay(ϕ, Δx, c)
    H = array_delay(delay_filter, τ*fs)

    X  = fft(x, 1)
    Tullio.@tullio HX[n,m] := H[n,m,k] * X[n,k]
    Hx = ifft(HX, 1)
    real.(Hx)[1:N,:] + σ*z_ϵ
end
