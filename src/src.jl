mkpath("data")

function up_to_date_charts(DO, loc="data/charts.csv")
    mkpath("data")

    DC=DataFrame(:date=>findmin(DO.inception_date)[1]:Day(1):Dates.today());

    first_day=DC.date[1]
    last_day=DC.date[end]
    
    i=1
    while i<=size(DO,1)
        tk=DO.isin[i]
        try 
            chart=JEft.load_chart(tk)["relative_with_reinvested_dividends"]
            
            dates=Date.(pyconvert(Vector,chart.index))
            vals=100 .+pyconvert(Vector,chart.values) 
            
            pad_below=Dates.value(dates[1]-first_day)
            pad_above=Dates.value(last_day-dates[end])

            if pad_below>=0
                insertcols!(DC,tk=>cat(fill(missing,pad_below),vals,fill(missing,pad_above); dims=1))
            else
                insertcols!(DC,tk=>cat(vals[1-pad_below:end],fill(missing,pad_above); dims=1))
            end
            
            println("$tk is loaded $i/$(size(DO,1))")
            i+=1
        catch e
            if e==InterruptException()
                throw(e)
            else
                println(e)
                println("Will sleep for 5 minutes and try again.")
                sleep(60*5)
            end
        end
    end

    CSV.write(loc,DC)

    return DC
end
function dist_name(nm::String)
    rules=(r"\(Acc\)$" => "(Dist)",r"\(acc\)$" => "(dist)", r"Acc$" => "Dist",  r"acc$" => "dist", r"A-acc$" => "A-dis", r"ETF$" => "ETF Distribution", r"ETF$" => "ETF Dist", r"Accumulating$"=>"Distributing", r"1C&"=>"1D")
    replace(nm, rules...)
end;


percent_return(vals,shift)=(vals[1+shift:end]-vals[1:end-shift])./vals[1:end-shift]

nrm(λ,k)=(1-λ^k)/(1-λ)

function weighted_mean(v,shift, half_life)
    λ=(1/2)^(1/(shift*half_life))
    v=collect(skipmissing(v))
    weights=reverse(λ.^(0:length(v)-1)./nrm(λ,length(v))) # we do reverse here as we want to give the highest weight to the most recent data    
    return sum(weights.*v)
end;

function get_R_RR_cor(DC,shift,half_life)    
    pr(x)=percent_return(x,shift)
    wm(x)=weighted_mean(x,shift,half_life)


    DC_no_missing=dropmissing(DC) # we remove the rows with missing elements for the covariance matrix computation

    returns_mat=Matrix(mapcols(pr,DC_no_missing[!,Not("date")]))

    target = DiagonalUnitVariance()
    shrinkage=:lw
    method = LinearShrinkage(target, shrinkage) # we use so called shrinkage estimators for the covariance matrix. This supposed to behave a bit more stable than the usual formula. 

    RR=cov(method, returns_mat)

    S=diagm(sqrt.(diag(RR)).^(-1))
    cor=S*RR*S

    R=Matrix(mapcols(wm∘pr,DC[!,Not("date")]))[:];

    return R, RR, cor
end;