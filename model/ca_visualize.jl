# Visualization utilities for 1D CA rule inference

module CAVisualize

using Plots
using Plots: RGB, cgrad
using Printf
using Random

using ..CARuleAST

export visualize_ca_evolution, visualize_rule_comparison
export visualize_posterior_predictions, create_inference_animation
export plot_posterior_rule_distribution
export compute_posterior_predictions, create_posterior_evolution_mp4
export create_uncertainty_colormap, render_prediction_frame


function visualize_ca_evolution(rows::Vector{<:AbstractVector{Bool}};
                                 title::String="CA Evolution",
                                 show_grid::Bool=false)
    n_steps = length(rows)
    width = length(rows[1])
    
    grid = zeros(Int, n_steps, width)
    for (t, row) in enumerate(rows)
        for (i, cell) in enumerate(row)
            grid[t, i] = cell ? 1 : 0
        end
    end
    
    plt = heatmap(grid,
                  color=:grays,
                  aspect_ratio=:equal,
                  xlabel="Cell Position",
                  ylabel="Time Step",
                  title=title,
                  yflip=true,
                  colorbar=false,
                  size=(max(400, width * 8), max(300, n_steps * 8)))
    
    if show_grid
        vline!(0.5:1:width+0.5, color=:lightgray, alpha=0.3, label=nothing)
        hline!(0.5:1:n_steps+0.5, color=:lightgray, alpha=0.3, label=nothing)
    end
    
    return plt
end

function visualize_ca_with_errors(true_rows::Vector{<:AbstractVector{Bool}},
                                   pred_rows::Vector{<:AbstractVector{Bool}};
                                   title::String="Prediction vs Truth")
    n_steps = length(true_rows)
    width = length(true_rows[1])
    
    grid = zeros(3, n_steps, width)
    
    for t in 1:n_steps
        for i in 1:width
            true_val = true_rows[t][i]
            pred_val = pred_rows[t][i]
            
            if true_val == pred_val
                val = true_val ? 1.0 : 0.0
                grid[1, t, i] = val
                grid[2, t, i] = val
                grid[3, t, i] = val
            else
                grid[1, t, i] = 1.0
                grid[2, t, i] = 0.0
                grid[3, t, i] = 0.0
            end
        end
    end
    
    img = permutedims(grid, (1, 3, 2))
    
    plt = plot(title=title,
               xlabel="Cell Position",
               ylabel="Time Step",
               size=(max(400, width * 8), max(300, n_steps * 8)))
    
    error_grid = zeros(Int, n_steps, width)
    for t in 1:n_steps
        for i in 1:width
            true_val = true_rows[t][i]
            pred_val = pred_rows[t][i]
            if true_val != pred_val
                error_grid[t, i] = 2
            elseif true_val
                error_grid[t, i] = 1
            else
                error_grid[t, i] = 0
            end
        end
    end
    
    colors = [:black, :white, :red]
    plt = heatmap(error_grid,
                  color=cgrad(colors, 3, categorical=true),
                  aspect_ratio=:equal,
                  xlabel="Cell Position",
                  ylabel="Time Step",
                  title=title,
                  yflip=true,
                  colorbar=false,
                  clims=(0, 2),
                  size=(max(400, width * 8), max(300, n_steps * 8)))
    
    return plt
end

export visualize_ca_with_errors


function visualize_rule_comparison(rule1::RuleNode,
                                   rule2::Union{RuleNode, Nothing},
                                   initial_row::Vector{Bool};
                                   num_steps::Int=30,
                                   labels::Tuple{String,String}=("Rule 1", "Rule 2"))
    rows1 = simulate_ca(rule1, initial_row, num_steps)
    
    if rule2 !== nothing
        rows2 = simulate_ca(rule2, initial_row, num_steps)
        p1 = visualize_ca_evolution(rows1, title=labels[1])
        p2 = visualize_ca_evolution(rows2, title=labels[2])
        plt = plot(p1, p2, layout=(1, 2), size=(800, 400))
    else
        plt = visualize_ca_evolution(rows1, title=labels[1])
    end
    
    return plt
end

function simulate_ca(rule::RuleNode, initial_row::Vector{Bool}, num_steps::Int)
    width = length(initial_row)
    rows = Vector{Vector{Bool}}(undef, num_steps + 1)
    rows[1] = copy(initial_row)
    
    for t in 1:num_steps
        rows[t + 1] = apply_rule_to_row(rule, rows[t])
    end
    
    return rows
end

function apply_rule_to_row(rule::RuleNode, row::Vector{Bool})
    width = length(row)
    new_row = Vector{Bool}(undef, width)
    
    for i in 1:width
        left = row[mod1(i - 1, width)]
        center = row[i]
        right = row[mod1(i + 1, width)]
        new_row[i] = eval_rule(rule, left, center, right)
    end
    
    return new_row
end

export simulate_ca, apply_rule_to_row


function visualize_posterior_predictions(rules::Vector{<:RuleNode},
                                          initial_row::Vector{Bool};
                                          true_rows::Union{Vector{<:AbstractVector{Bool}}, Nothing}=nothing,
                                          num_steps::Int=20,
                                          num_samples::Int=5,
                                          title::String="Posterior Predictions")
    n_rules = length(rules)
    num_samples = min(num_samples, n_rules)
    
    sample_indices = randperm(n_rules)[1:num_samples]
    
    plots = []
    
    if true_rows !== nothing
        p_true = visualize_ca_evolution(true_rows[1:min(num_steps+1, length(true_rows))],
                                        title="Ground Truth")
        push!(plots, p_true)
    end
    
    for (i, idx) in enumerate(sample_indices)
        rows = simulate_ca(rules[idx], initial_row, num_steps)
        rule_num = rule_to_wolfram_number(rules[idx])
        p = visualize_ca_evolution(rows, title="Sample $i (Rule $rule_num)")
        push!(plots, p)
    end
    
    n_plots = length(plots)
    ncols = min(3, n_plots)
    nrows = ceil(Int, n_plots / ncols)
    
    plt = plot(plots..., 
               layout=(nrows, ncols),
               size=(ncols * 300, nrows * 250),
               plot_title=title)
    
    return plt
end


function plot_posterior_rule_distribution(rules::Vector{<:RuleNode};
                                          true_rule_number::Union{Int, Nothing}=nothing,
                                          title::String="Posterior Rule Distribution")
    rule_numbers = [rule_to_wolfram_number(r) for r in rules]
    
    counts = Dict{Int, Int}()
    for rn in rule_numbers
        counts[rn] = get(counts, rn, 0) + 1
    end
    
    sorted_rules = sort(collect(counts), by=x -> -x[2])
    
    n_top = min(20, length(sorted_rules))
    top_rules = sorted_rules[1:n_top]
    
    rule_labels = [string(r[1]) for r in top_rules]
    rule_counts = [r[2] for r in top_rules]
    
    colors = [:steelblue for _ in 1:n_top]
    
    if true_rule_number !== nothing
        for (i, (rn, _)) in enumerate(top_rules)
            if rn == true_rule_number
                colors[i] = :red
                break
            end
        end
    end
    
    plt = bar(rule_labels, rule_counts,
              color=colors,
              xlabel="Wolfram Rule Number",
              ylabel="Count",
              title=title,
              legend=false,
              rotation=45,
              size=(600, 400))
    
    if true_rule_number !== nothing
        true_count = get(counts, true_rule_number, 0)
        pct = 100 * true_count / length(rules)
        annotate!(plt, [(length(rule_labels)/2, maximum(rule_counts) * 0.9, 
                        text("True rule ($true_rule_number): $(round(pct, digits=1))%", 10, :red))])
    end
    
    return plt
end


function create_inference_animation(history::Vector{<:Vector{<:RuleNode}},
                                     true_rows::Vector{<:AbstractVector{Bool}};
                                     initial_row::Union{Vector{Bool}, Nothing}=nothing,
                                     true_rule_number::Union{Int, Nothing}=nothing,
                                     fps::Int=2,
                                     filename::String="figures/ca_inference.gif")
    T = length(history)
    
    if initial_row === nothing
        initial_row = Vector{Bool}(true_rows[1])
    end
    
    println("Creating animation with $T frames...")
    
    anim = @animate for t in 1:T
        rules = history[t]
        
        rule_numbers = [rule_to_wolfram_number(r) for r in rules]
        counts = Dict{Int, Int}()
        for rn in rule_numbers
            counts[rn] = get(counts, rn, 0) + 1
        end
        map_rule_num = argmax(counts)
        
        map_idx = findfirst(r -> rule_to_wolfram_number(r) == map_rule_num, rules)
        map_rule = rules[map_idx]
        
        pred_rows = simulate_ca(map_rule, initial_row, length(true_rows) - 1)
        
        p1 = visualize_ca_evolution(true_rows, title="Ground Truth")
        p2 = visualize_ca_evolution(pred_rows, title="MAP Prediction (Rule $map_rule_num)")
        
        hline!(p1, [t + 0.5], color=:red, linewidth=2, label="t=$t")
        hline!(p2, [t + 0.5], color=:red, linewidth=2, label=nothing)
        
        p3 = plot_posterior_rule_distribution(rules; 
                                              true_rule_number=true_rule_number,
                                              title="Posterior at t=$t")
        
        plot(p1, p2, p3, layout=(1, 3), size=(1200, 400),
             plot_title="CA Rule Inference: t = $t / $T")
    end
    
    if endswith(filename, ".mp4")
        mp4(anim, filename, fps=fps)
    else
        gif(anim, filename, fps=fps)
    end
    
    println("Saved animation to $filename")
    
    return anim
end


function print_posterior_summary(rules::Vector{<:RuleNode};
                                  true_rule_number::Union{Int, Nothing}=nothing)
    rule_numbers = [rule_to_wolfram_number(r) for r in rules]
    
    counts = Dict{Int, Int}()
    for rn in rule_numbers
        counts[rn] = get(counts, rn, 0) + 1
    end
    
    sorted_rules = sort(collect(counts), by=x -> -x[2])
    
    println("Posterior Summary ($(length(rules)) samples):")
    println("-" ^ 40)
    
    println("Top 5 rules:")
    for (i, (rn, count)) in enumerate(sorted_rules[1:min(5, length(sorted_rules))])
        pct = 100 * count / length(rules)
        println(@sprintf("  %d. Rule %3d: %4d samples (%.1f%%)", i, rn, count, pct))
    end
    
    if true_rule_number !== nothing
        true_count = get(counts, true_rule_number, 0)
        true_pct = 100 * true_count / length(rules)
        
        rank = findfirst(x -> x[1] == true_rule_number, sorted_rules)
        rank_str = rank === nothing ? "not in top" : "rank $rank"
        
        println()
        println("True rule ($true_rule_number): $true_count samples ($(round(true_pct, digits=1))%), $rank_str")
    end
    
    println()
    println("Unique rules in posterior: $(length(counts))")
end

export print_posterior_summary


function compute_posterior_predictions(rules::AbstractVector{<:RuleNode},
                                       current_row::Vector{Bool},
                                       num_future_steps::Int)
    N = length(rules)
    width = length(current_row)
    
    probs = zeros(Float64, num_future_steps, width)
    
    for rule in rules
        row = copy(current_row)
        for t in 1:num_future_steps
            row = apply_rule_to_row(rule, row)
            for i in 1:width
                probs[t, i] += row[i] ? 1.0 : 0.0
            end
        end
    end
    
    probs ./= N
    
    return probs
end

function create_uncertainty_colormap()
    return cgrad([
        RGB(1.0, 1.0, 1.0),
        RGB(1.0, 0.6, 0.2),
        RGB(1.0, 0.3, 0.1),
        RGB(0.5, 0.15, 0.05),
        RGB(0.0, 0.0, 0.0)
    ])
end

function build_combined_display(observed_rows::Vector{<:AbstractVector{Bool}},
                                 pred_probs::Matrix{Float64})
    n_observed = length(observed_rows)
    n_future = size(pred_probs, 1)
    width = length(observed_rows[1])
    
    total_rows = n_observed + n_future
    combined = zeros(Float64, total_rows, width)
    
    for (t, row) in enumerate(observed_rows)
        for i in 1:width
            combined[t, i] = row[i] ? 1.0 : 0.0
        end
    end
    
    for t in 1:n_future
        for i in 1:width
            combined[n_observed + t, i] = pred_probs[t, i]
        end
    end
    
    return combined
end

function render_prediction_frame(t::Int,
                                  rules::AbstractVector{<:RuleNode},
                                  true_rows::Vector{<:AbstractVector{Bool}};
                                  num_future_display::Int=20,
                                  show_ground_truth::Bool=true)
    n_observed = t + 1
    observed_rows = [Vector{Bool}(true_rows[i]) for i in 1:n_observed]
    current_row = observed_rows[end]
    width = length(current_row)
    
    pred_probs = compute_posterior_predictions(rules, current_row, num_future_display)
    combined = build_combined_display(observed_rows, pred_probs)
    total_rows = size(combined, 1)
    
    cmap = create_uncertainty_colormap()
    
    plt = heatmap(combined,
                  color=cmap,
                  aspect_ratio=:equal,
                  xlabel="Cell Position",
                  ylabel="Time Step",
                  yflip=true,
                  colorbar=true,
                  colorbar_title="P(alive)",
                  clims=(0, 1),
                  size=(max(600, width * 12), max(500, total_rows * 10)))
    
    hline!([n_observed + 0.5], 
           color=:red, 
           linewidth=3, 
           linestyle=:solid,
           label=nothing)
    
    annotate!(width + 1.5, n_observed + 0.5, 
              text("← observed | predicted →", 8, :red, :left))
    
    if show_ground_truth && length(true_rows) > n_observed
        n_future_available = min(num_future_display, length(true_rows) - n_observed)
        
        for dt in 1:n_future_available
            future_row = true_rows[n_observed + dt]
            y_pos = n_observed + dt
            
            for i in 1:width
                if future_row[i]
                    scatter!([i], [y_pos], 
                            marker=:diamond, 
                            markersize=2,
                            markercolor=:lime,
                            markerstrokecolor=:black,
                            markerstrokewidth=0.5,
                            label=nothing)
                end
            end
        end
    end
    
    return plt
end

function create_posterior_evolution_mp4(history::Vector{<:Vector{<:RuleNode}},
                                         true_rows::Vector{<:AbstractVector{Bool}};
                                         num_future_display::Int=20,
                                         fps::Int=4,
                                         filename::String="figures/posterior_evolution.mp4",
                                         show_ground_truth::Bool=true,
                                         skip_frames::Int=1)
    T = length(history)
    width = length(true_rows[1])
    
    println("Creating posterior evolution animation...")
    println("  Total observations: $T")
    println("  Future steps shown: $num_future_display")
    println("  Output: $filename")
    
    frames_to_render = 1:skip_frames:T
    n_frames = length(frames_to_render)
    println("  Frames to render: $n_frames")
    
    anim = @animate for (frame_idx, t) in enumerate(frames_to_render)
        rules = history[t]
        
        p_main = render_prediction_frame(t, rules, true_rows;
                                         num_future_display=num_future_display,
                                         show_ground_truth=show_ground_truth)
        
        title!(p_main, "Posterior Predictive Distribution: t = $t / $T observations")
        
        rule_numbers = [rule_to_wolfram_number(r) for r in rules]
        counts = Dict{Int, Int}()
        for rn in rule_numbers
            counts[rn] = get(counts, rn, 0) + 1
        end
        map_rule_num = argmax(counts)
        map_pct = round(100 * counts[map_rule_num] / length(rules), digits=1)
        n_unique = length(counts)
        
        p_dist = plot_posterior_rule_distribution(rules;
                                                   title="Rule Distribution (n=$n_unique unique)")
        
        layout = @layout [a{0.7w} b{0.3w}]
        plot(p_main, p_dist, 
             layout=layout,
             size=(1200, 600),
             plot_title="MAP: Rule $map_rule_num ($map_pct%)")
    end
    
    println("  Encoding video...")
    if endswith(filename, ".mp4")
        mp4(anim, filename, fps=fps)
    elseif endswith(filename, ".gif")
        gif(anim, filename, fps=fps)
    else
        mp4(anim, filename * ".mp4", fps=fps)
    end
    
    println("  Saved: $filename")
    
    return anim
end

function visualize_prediction_uncertainty(rules::Vector{<:RuleNode},
                                          current_row::Vector{Bool};
                                          num_future::Int=20,
                                          title::String="Prediction Uncertainty")
    probs = compute_posterior_predictions(rules, current_row, num_future)
    uncertainty = -probs .* log2.(probs .+ 1e-10) .- (1 .- probs) .* log2.(1 .- probs .+ 1e-10)
    
    cmap = create_uncertainty_colormap()
    
    p1 = heatmap(probs,
                 color=cmap,
                 aspect_ratio=:equal,
                 xlabel="Cell Position",
                 ylabel="Future Time Step",
                 title="P(alive)",
                 yflip=true,
                 colorbar=true,
                 clims=(0, 1))
    
    p2 = heatmap(uncertainty,
                 color=:hot,
                 aspect_ratio=:equal,
                 xlabel="Cell Position",
                 ylabel="Future Time Step",
                 title="Uncertainty (bits)",
                 yflip=true,
                 colorbar=true,
                 clims=(0, 1))
    
    plt = plot(p1, p2, layout=(1, 2), size=(900, 400), plot_title=title)
    
    return plt
end

export visualize_prediction_uncertainty

end
