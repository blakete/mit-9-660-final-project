# SMC inference for 1D CA rule learning

using Random
using Printf
using Plots
using Statistics

include("ca_rule_ast.jl")
using .CARuleAST

include("ca_grammar_prior.jl")
using .CAGrammarPrior

include("ca_rejuvenation.jl")
using .CARejuvenation

include("ca_visualize.jl")
using .CAVisualize

using Gen


mutable struct RuleParticle
    rule::RuleNode
    log_weight::Float64
end


function log_likelihood_cell(rule::RuleNode, left::Bool, center::Bool, right::Bool,
                             observed::Bool; ε::Float64=0.01)
    predicted = eval_rule(rule, left, center, right)
    if predicted == observed
        return log(1 - ε)
    else
        return log(ε)
    end
end

function log_likelihood_transition(rule::RuleNode, row_old::Vector{Bool},
                                   row_new::Vector{Bool}; ε::Float64=0.01)
    @assert length(row_old) == length(row_new) "Row lengths must match"
    
    width = length(row_new)
    ll = 0.0
    
    for i in 1:width
        left = row_old[mod1(i - 1, width)]
        center = row_old[i]
        right = row_old[mod1(i + 1, width)]
        ll += log_likelihood_cell(rule, left, center, right, row_new[i]; ε=ε)
    end
    
    return ll
end

function log_likelihood_all(rule::RuleNode, rows::Vector{<:AbstractVector{Bool}},
                            t::Int; ε::Float64=0.01)
    ll = 0.0
    for i in 1:t
        ll += log_likelihood_transition(rule, Vector{Bool}(rows[i]), Vector{Bool}(rows[i+1]); ε=ε)
    end
    return ll
end


function generate_ca_data(rule_number::Int; width::Int=31, num_steps::Int=30,
                          initial::Union{Symbol, Vector{Bool}}=:single_center)
    @assert 0 <= rule_number <= 255 "Rule number must be 0-255"
    
    lookup = wolfram_rule_to_lookup_table(rule_number)
    
    if initial == :random
        initial_row = rand(Bool, width)
    elseif initial == :single_center
        initial_row = fill(false, width)
        initial_row[div(width, 2) + 1] = true
    else
        initial_row = initial
    end
    
    rows = Vector{Vector{Bool}}(undef, num_steps + 1)
    rows[1] = initial_row
    
    for t in 1:num_steps
        rows[t + 1] = Vector{Bool}(undef, width)
        for i in 1:width
            left = rows[t][mod1(i - 1, width)]
            center = rows[t][i]
            right = rows[t][mod1(i + 1, width)]
            rows[t + 1][i] = lookup[(left, center, right)]
        end
    end
    
    return rows, lookup
end

function generate_ca_data_from_tree(rule::RuleNode; width::Int=31, num_steps::Int=30,
                                    initial::Union{Symbol, Vector{Bool}}=:single_center)
    if initial == :random
        initial_row = rand(Bool, width)
    elseif initial == :single_center
        initial_row = fill(false, width)
        initial_row[div(width, 2) + 1] = true
    else
        initial_row = initial
    end
    
    return simulate_ca(rule, initial_row, num_steps)
end


function initialize_particles(num_particles::Int, p_leaf::Float64, max_depth::Int)
    particles = Vector{RuleParticle}(undef, num_particles)
    
    for i in 1:num_particles
        trace = Gen.simulate(sample_rule_tree, (p_leaf, max_depth))
        rule = Gen.get_retval(trace)
        particles[i] = RuleParticle(rule, 0.0)
    end
    
    return particles
end

function compute_ess(log_weights::Vector{Float64})
    max_lw = maximum(log_weights)
    weights = exp.(log_weights .- max_lw)
    weights ./= sum(weights)
    return 1.0 / sum(weights .^ 2)
end

function sample_categorical(weights::Vector{Float64}, n::Int)
    cumweights = cumsum(weights)
    total = cumweights[end]
    
    indices = Vector{Int}(undef, n)
    for i in 1:n
        u = rand() * total
        idx = searchsortedfirst(cumweights, u)
        indices[i] = clamp(idx, 1, length(weights))
    end
    return indices
end

function resample_particles(particles::Vector{RuleParticle})
    n = length(particles)
    log_weights = [p.log_weight for p in particles]
    
    max_lw = maximum(log_weights)
    weights = exp.(log_weights .- max_lw)
    weights ./= sum(weights)
    
    indices = sample_categorical(weights, n)
    
    new_particles = Vector{RuleParticle}(undef, n)
    for i in 1:n
        new_particles[i] = RuleParticle(particles[indices[i]].rule, 0.0)
    end
    
    return new_particles
end


function ca_rule_particle_filter(rows::Vector{<:AbstractVector{Bool}};
                                  num_particles::Int=1000,
                                  ε::Float64=0.01,
                                  p_leaf::Float64=0.5,
                                  max_depth::Int=5,
                                  num_rejuv_steps::Int=3,
                                  ess_threshold_ratio::Float64=0.5,
                                  verbose::Bool=true)
    T_total = length(rows) - 1
    ess_threshold = num_particles * ess_threshold_ratio
    
    if verbose
        println("Initializing $num_particles particles...")
    end
    particles = initialize_particles(num_particles, p_leaf, max_depth)
    
    history = Vector{Vector{RuleNode}}(undef, T_total)
    map_history = Vector{RuleNode}(undef, T_total)
    
    for t in 1:T_total
        if verbose && (t % 5 == 0 || t == 1)
            println("  Processing transition $t / $T_total")
        end
        
        row_old = Vector{Bool}(rows[t])
        row_new = Vector{Bool}(rows[t + 1])
        
        for p in particles
            ll = log_likelihood_transition(p.rule, row_old, row_new; ε=ε)
            p.log_weight += ll
        end
        
        log_weights = [p.log_weight for p in particles]
        ess = compute_ess(log_weights)
        
        if ess < ess_threshold
            if verbose
                println("    Resampling at t=$t (ESS=$(round(ess, digits=1)))")
            end
            particles = resample_particles(particles)
        end
        
        rows_so_far = rows[1:t+1]
        for p in particles
            function ll_fn(rule::RuleNode)
                return log_likelihood_all(rule, rows_so_far, t; ε=ε)
            end
            
            p.rule = rejuvenate_rule_with_likelihood(p.rule, ll_fn;
                                                      num_steps=num_rejuv_steps,
                                                      max_depth=max_depth)
            
            p.log_weight = log_likelihood_all(p.rule, rows_so_far, t; ε=ε)
            p.log_weight += log_prior_tree(p.rule, p_leaf)
        end
        
        history[t] = [p.rule for p in particles]
        best_idx = argmax([p.log_weight for p in particles])
        map_history[t] = particles[best_idx].rule
    end
    
    return history, map_history, particles
end


function analyze_posterior_structures(particles::Vector{RuleParticle})
    n = length(particles)
    
    sizes = [count_nodes(p.rule) for p in particles]
    n_atoms = [count_atoms(p.rule) for p in particles]
    rule_numbers = [rule_to_wolfram_number(p.rule) for p in particles]
    
    println("Posterior structure statistics:")
    @printf("  Tree size:     mean=%.1f, min=%d, max=%d\n", 
            mean(sizes), minimum(sizes), maximum(sizes))
    @printf("  Num atoms:     mean=%.1f, min=%d, max=%d\n",
            mean(n_atoms), minimum(n_atoms), maximum(n_atoms))
    
    unique_rules = unique(rule_numbers)
    println("  Unique rules:  $(length(unique_rules))")
    
    counts = Dict{Int, Int}()
    for rn in rule_numbers
        counts[rn] = get(counts, rn, 0) + 1
    end
    sorted = sort(collect(counts), by=x -> -x[2])
    
    println("  Top 3 rules:")
    for (i, (rn, count)) in enumerate(sorted[1:min(3, length(sorted))])
        pct = 100 * count / n
        @printf("    %d. Rule %3d: %.1f%%\n", i, rn, pct)
    end
end

function find_map_particle(particles::Vector{RuleParticle})
    best_idx = argmax([p.log_weight for p in particles])
    return particles[best_idx]
end

function print_rule(rule::RuleNode; label::String="")
    if !isempty(label)
        println(label)
    end
    println("  Expression: ", rule_to_string(rule))
    println("  Verbose:    ", rule_to_expr_string(rule))
    println("  Wolfram #:  ", rule_to_wolfram_number(rule))
    println("  Size:       ", count_nodes(rule), " nodes")
end


function main()
    Random.seed!(42)
    
    println("="^60)
    println("CA Rule Inference via Bayesian Program Synthesis")
    println("="^60)
    println()
    
    true_rule_number = 110
    
    println("Generating CA data from Wolfram Rule $true_rule_number...")
    rows, _ = generate_ca_data(true_rule_number; width=8, num_steps=30, initial=:random)
    
    println("CA width: $(length(rows[1])), steps: $(length(rows) - 1)")
    println()
    
    num_particles = 256
    println("Running SMC with $num_particles particles...")
    println()
    
    history, map_history, final_particles = ca_rule_particle_filter(
        rows;
        num_particles=num_particles,
        ε=0.05,
        p_leaf=0.5,
        max_depth=5,
        num_rejuv_steps=5,
        verbose=true
    )
    
    println()
    println("="^60)
    println("Results")
    println("="^60)
    println()
    
    analyze_posterior_structures(final_particles)
    println()
    
    map_particle = find_map_particle(final_particles)
    print_rule(map_particle.rule, label="MAP rule:")
    @printf("  Log weight: %.3f\n", map_particle.log_weight)
    
    map_rule_num = rule_to_wolfram_number(map_particle.rule)
    println()
    if map_rule_num == true_rule_number
        println("✓ MAP rule matches ground truth!")
    else
        println("✗ MAP rule ($map_rule_num) differs from ground truth ($true_rule_number)")
        correct_count = count(p -> rule_to_wolfram_number(p.rule) == true_rule_number, 
                              final_particles)
        println("  Particles with correct rule: $correct_count / $(length(final_particles))")
    end
    
    println()
    println("Generating visualizations...")
    
    mkpath("figures")
    
    plt1 = visualize_ca_evolution(rows; title="Ground Truth: Rule $true_rule_number")
    savefig(plt1, "figures/ca_ground_truth.png")
    println("Saved: figures/ca_ground_truth.png")
    
    initial_row = Vector{Bool}(rows[1])
    map_rows = simulate_ca(map_particle.rule, initial_row, length(rows) - 1)
    plt2 = visualize_ca_evolution(map_rows; title="MAP Prediction: Rule $map_rule_num")
    savefig(plt2, "figures/ca_map_prediction.png")
    println("Saved: figures/ca_map_prediction.png")
    
    final_rules = [p.rule for p in final_particles]
    plt3 = plot_posterior_rule_distribution(final_rules;
                                            true_rule_number=true_rule_number,
                                            title="Posterior Rule Distribution")
    savefig(plt3, "figures/ca_posterior_distribution.png")
    println("Saved: figures/ca_posterior_distribution.png")
    
    plt4 = visualize_rule_comparison(
        make_rule_110(),
        map_particle.rule,
        initial_row;
        num_steps=10,
        labels=("Ground Truth", "MAP Estimate (Rule $map_rule_num)")
    )
    savefig(plt4, "figures/ca_comparison.png")
    println("Saved: figures/ca_comparison.png")
    
    println()
    print_posterior_summary(final_rules; true_rule_number=true_rule_number)
    
    println()
    println("Creating inference animation...")
    create_inference_animation(history, rows;
                               initial_row=initial_row,
                               true_rule_number=true_rule_number,
                               fps=3,
                               filename="figures/ca_inference.mp4")
    
    println()
    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
