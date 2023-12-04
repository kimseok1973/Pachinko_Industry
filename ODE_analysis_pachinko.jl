using CSV,StatsPlots,DataFrames
using ModelingToolkit
using DifferentialEquations

@variables t P(t) V(t) N(t)
@parameters  r e b tevent
D = Differential(t)

eqs = [ D(P) ~ e * ( V - P )  - r * P
        D(N) ~ r * P  - e *( V - P) + D(V)
        D(V) ~ b        ]

# データセットを作成する
#=
dx = CSV.read(joinpath("p-datasets2.csv"),DataFrame)
for (i,e) in enumerate(names(dx))
    print("$i : $e ,")
end
eq = zeros(31) ; eq[23:end] .= 1.0
dx.eq = eq ;
#dx.パチンコ参加人口 = dx.パチンコ参加人口 .* 10 ;
dx

Pu = dx.パチンコ参加人口[6:end]
Vu = dx.総人口[6:end]
year = dx.年[6:end]
df=DataFrame(:Year => year, :Player=>Pu, :Population=>Vu)
"""
=#

#CSV.write("ODE_analysis_pachinko.df.csv",df)
df = CSV.read("ODE_analysis_pachinko.df.csv",DataFrame)
Pu = df.Player
Vu = df.Population
year = df.Year
plot(year, Pu)

"""
    実数時間ｔにおける観測データを返す関数
    serial(3 ; data=ys)
    ys[3]を返す

    tが0の時は無条件にt＝1のデータを返す
"""
function serial(t; data=data) ::Float64
    #t = t +rand(Normal(0.1,0.1))
    m = length(data)
    if t < 1
        nn = data[1]  + randn()
    elseif t > m
        nn = data[end] + randn()
    else 
        #実数と小数点以下とに分ける
        #f, n = divrem(t, 1) divremだとなぜかエラーになる　原因わからず
        f = floor(t)
        n = t - f
        #実数に対応するデータをff、ff+1をffnに入れる
        #ffnはtがデータ時間より長い場合、無条件に最後のデータを返す
        if f >= length(data)
            nn = data[end]  + + randn()
            #nn = (data[t-1  |> Int ] + data[t+1  |> Int ])/2
        else
            ff = data[ f |> Int]
            ffn =data[(f+1) |> Int]
            nn = ff + (ffn - ff) * n
        end
    end
    return nn
end

pu_func(t) = serial(t ; data=Pu)
vu_func(t) = serial(t ; data=Vu)

begin
    xs = 1:0.2:50
    plot(xs,[pu_func(t) for t = xs],label="function data")
    plot!(1:20, Pu[1:20],lw=2,label="real data")
    vline!([2,16] ,ls=:dash,label="catastrophe")
end

## AutoDiffenceのZygoteをつかってデータ関数を微分してみる

using Zygote

dvu(τ)=gradient(x->vu_func(x), τ)[1]
begin
    xs = 1:0.1:30
    d_vu =[dvu(t) for t in xs]
    plot([e == nothing ? 0 : e for e in d_vu])
end

@register_symbolic pu_func(t)
@register_symbolic vu_func(t)
@register_symbolic dvu(t)

eqs_v = [  D(P) ~ e * ( V - P )  - r * P
           D(N) ~ r * P  - e *( V - P) + D(V)
           D(V) ~ dvu(t) ]

@named sysv = ODESystem(eqs_v, t)
sysv = structural_simplify(sysv)

begin
    u0 = [P => Pu[1] ,N => Vu[1] - Pu[1] , V => Vu[1]]
    ps = [e => 0.03, r=> 0.4]
    tspan=(1,length(year)-1) # 短くしないとエラー
    probv = ODEProblem(sysv, u0, tspan, ps)
    solv = solve(probv)
    plot(solv, idxs=[P],label="$ps")

    ps2=[e=> 0.05, r=>0.4]
    rprob=remake(probv ; u0 = u0 , p=ps2) 
    rsolv=solve(rprob, saveat=collect(1:0.1:25.5))
    plot!(rsolv, idxs=[P], label="$ps")
end

## Find Parameter 
using Turing, Distributions

@model function findprm(prob, Px, Vx, ts) # findprm(Pu, Vu, probv)
    n = length(Px)
    entry  ~ truncated(TDist(3), 1e-3, 1 - 1e-3)
    retire ~ truncated(TDist(3), 1e-3, 1 - 1e-3 )
    s ~ InverseGamma(2,3)
    
    prob = remake(prob ; 
            u0= [ P => Px[1], N => Vx[1] - Px[1], V => Vx[1] ],
            p = [ e => entry, r => retire],
            tspan = ts
    )
    solv = solve(prob, saveat=collect(ts[1]:ts[end]) )
    
    for i = 1:n-1 #　nまでやるとエラー
        solt = solv.u[i][1]
        Px[i] ~ Normal(solt, s) 
        #Px[i] ~ Normal(solv.u[i][1], s) 
    end
    return (; Ps = [])
end

ts = ( 1,length(year)-0.1 ) #(1.0, 25.0 )
model = findprm(probv, Pu, Vu, ts)
chain = sample(model, NUTS(.64), 2000, progress=true)
ts_0 = get(chain, [:entry, :retire]) |> t->  (; e = t.entry |> mean, r = t.retire |> mean)
begin
    rsol =solve( remake( probv, p = [e => ts_0.e, r => ts_0.r]) , saveat=collect(ts[1]:ts[end]))
    plot(rsol, idxs=[P])
    scatter!(0:ts[2], Pu)
end