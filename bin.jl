R,RR,cor=get_R_RR_cor(DC_acc_cr, shift, half_life)

function entropy(w)
    -sum((w).*log.(w))
end
function sharpe_ratio(w) 
    return (dot(R,w)-risk_free_return)/sqrt(dot(w,RR*w))
end
function objective(w)
    return sharpe_ratio(w)+entropy_factor*entropy(w)
end

model=Model(NLopt.Optimizer)
@variable(model, 0<=w[1:length(R)]<=1)
@objective(model,Max,objective(w))
@constraint(model, sum(w)==1)
@constraint(model, sum(w[USD_filter])<=USD_bound)


set_attribute(model, "algorithm", :LD_SLSQP)
set_attribute(model, "constrtol_abs", 1e-8)
set_attribute(model, "ftol_rel", 1e-3)

w0=rand(length(R))
w0/=sum(w0)
set_start_value.(w, w0)

JuMP.optimize!(model)

if !is_solved_and_feasible(model)
    throw("Optimizer did not find a solution!")
end

println("termination status: ", termination_status(model))

w_opt=value.(w)
sr0=dot(R,w_opt)/sqrt(dot(w_opt,RR*w_opt))

println(raw_status(model), "| objective: ", objective_value(model), " entropy: ", entropy(w_opt), " return: ", dot(R,w_opt), " risk: ", sqrt(dot(w_opt,RR*w_opt)), " return/risk ratio: ", sr0)

if lowest_share!=0
    println("Now, the optimization problem ill be restricted to the tickers with more than $(lowest_share*100)%")
    low_share_filter=w_opt.<=lowest_share
    @constraint(model, sum(w[low_share_filter])<=1e-10)
    
    set_start_value.(w, w_opt)
    JuMP.optimize!(model)

    if !is_solved_and_feasible(model)
        throw("Optimizer did not find a solution!")
    end

    println("termination status: ", termination_status(model))
    w_opt=value.(w)
    sr0=dot(R,w_opt)/sqrt(dot(w_opt,RR*w_opt))

    println(raw_status(model), "| objective: ", objective_value(model), " entropy: ", entropy(w_opt), " return: ", dot(R,w_opt), " risk: ", sqrt(dot(w_opt,RR*w_opt)), " return/risk ratio: ", sr0)
end