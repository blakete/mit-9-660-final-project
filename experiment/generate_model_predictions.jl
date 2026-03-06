# Run Bayesian SMC inference and compute model predictions for human experiment

using JSON
using Dates
using Printf
using Statistics
using Random

include(joinpath(@__DIR__, "..", "model", "ca_smc.jl"))

const CA_WIDTH = 17
const TIME_STEPS_TO_TEST = 1:6
const RANDOM_SEED = 42

const NUM_PARTICLES = 20
const EPSILON = 0.20
const P_LEAF = 0.5
const MAX_DEPTH = 5
const NUM_REJUV_STEPS = 3
const DECISION_TEMPERATURE = 3.0

const RULES_TO_TEST = [
    (rule=32, config_path="rule_032/run_1_config.json"),
    (rule=5, config_path="rule_005/run_1_config.json"),
    (rule=108, config_path="rule_108/run_1_config.json"),
    (rule=30, config_path="rule_030/run_1_config.json"),
    (rule=60, config_path="rule_060/run_1_config.json"),
    (rule=54, config_path="rule_054/run_3_config.json"),
    (rule=110, config_path="rule_110/run_1_config.json"),
]


function load_initial_row(config_path::String)::Vector{Bool}
    full_path = joinpath(@__DIR__, "..", "zoo_of_rules", config_path)
    config = JSON.parsefile(full_path)
    full_row = Bool.(config["initial_row"])
    
    original_width = length(full_row)
    if original_width <= CA_WIDTH
        return full_row
    end
    
    start_idx = div(original_width - CA_WIDTH, 2) + 1
    end_idx = start_idx + CA_WIDTH - 1
    return full_row[start_idx:end_idx]
end

function wolfram_rule_to_lookup(rule_number::Int)::Dict{Tuple{Bool,Bool,Bool}, Bool}
    neighborhoods = [
        (true, true, true),
        (true, true, false),
        (true, false, true),
        (true, false, false),
        (false, true, true),
        (false, true, false),
        (false, false, true),
        (false, false, false)
    ]
    
    table = Dict{Tuple{Bool,Bool,Bool}, Bool}()
    for (i, neigh) in enumerate(neighborhoods)
        bit_position = 8 - i
        table[neigh] = (rule_number >> bit_position) & 1 == 1
    end
    return table
end

function apply_wolfram_rule(rule_number::Int, row::Vector{Bool})::Vector{Bool}
    lookup = wolfram_rule_to_lookup(rule_number)
    width = length(row)
    new_row = Vector{Bool}(undef, width)
    
    for i in 1:width
        left = row[mod1(i - 1, width)]
        center = row[i]
        right = row[mod1(i + 1, width)]
        new_row[i] = lookup[(left, center, right)]
    end
    
    return new_row
end

function generate_ca_evolution(rule_number::Int, initial_row::Vector{Bool}, num_steps::Int)::Vector{Vector{Bool}}
    rows = Vector{Vector{Bool}}(undef, num_steps + 1)
    rows[1] = copy(initial_row)
    
    for t in 1:num_steps
        rows[t + 1] = apply_wolfram_rule(rule_number, rows[t])
    end
    
    return rows
end

function apply_rule_tree(rule::CARuleAST.RuleNode, row::Vector{Bool})::Vector{Bool}
    width = length(row)
    new_row = Vector{Bool}(undef, width)
    
    for i in 1:width
        left = row[mod1(i - 1, width)]
        center = row[i]
        right = row[mod1(i + 1, width)]
        new_row[i] = CARuleAST.eval_rule(rule, left, center, right)
    end
    
    return new_row
end

function compute_candidate_probability(particles::Vector{RuleParticle}, 
                                        current_row::Vector{Bool},
                                        candidate_row::Vector{Bool})::Float64
    n_match = 0
    
    for p in particles
        predicted_row = apply_rule_tree(p.rule, current_row)
        if predicted_row == candidate_row
            n_match += 1
        end
    end
    
    return n_match / length(particles)
end

function apply_decision_noise(probs::Vector{Float64}, temperature::Float64)::Vector{Float64}
    if all(p -> p ≈ 0.0, probs)
        return fill(1.0 / length(probs), length(probs))
    end
    
    exponent = 1.0 / temperature
    noisy_probs = (probs .+ 1e-10) .^ exponent
    return noisy_probs ./ sum(noisy_probs)
end

function compute_all_candidate_probabilities(particles::Vector{RuleParticle},
                                              current_row::Vector{Bool},
                                              candidate_rules::Vector{Int})::Vector{Float64}
    raw_probs = Float64[]
    
    for rule_num in candidate_rules
        candidate_row = apply_wolfram_rule(rule_num, current_row)
        prob = compute_candidate_probability(particles, current_row, candidate_row)
        push!(raw_probs, prob)
    end
    
    decision_probs = apply_decision_noise(raw_probs, DECISION_TEMPERATURE)
    return decision_probs
end


function generate_predictions_for_rule(rule_number::Int, 
                                        initial_row::Vector{Bool},
                                        trial_options::Dict{Int, Vector{Int}})::Dict{String, Any}
    max_t = maximum(TIME_STEPS_TO_TEST) + 1
    true_rows = generate_ca_evolution(rule_number, initial_row, max_t)
    
    println("  Initializing $NUM_PARTICLES particles...")
    particles = initialize_particles(NUM_PARTICLES, P_LEAF, MAX_DEPTH)
    
    predictions = Dict{String, Any}()
    
    for t in TIME_STEPS_TO_TEST
        trial_id = @sprintf("rule_%03d_t%d", rule_number, t)
        println("    Processing $trial_id...")
        
        current_row = true_rows[t]
        candidate_rules = trial_options[t]
        
        probs = compute_all_candidate_probabilities(particles, current_row, candidate_rules)
        
        predictions[trial_id] = Dict(
            "candidate_probabilities" => probs,
            "candidate_rules" => candidate_rules
        )
        
        if t < maximum(TIME_STEPS_TO_TEST)
            row_old = Vector{Bool}(true_rows[t])
            row_new = Vector{Bool}(true_rows[t + 1])
            
            for p in particles
                ll = log_likelihood_transition(p.rule, row_old, row_new; ε=EPSILON)
                p.log_weight += ll
            end
            
            log_weights = [p.log_weight for p in particles]
            ess = compute_ess(log_weights)
            ess_threshold = NUM_PARTICLES * 0.5
            
            if ess < ess_threshold
                particles = resample_particles(particles)
            end
            
            rows_so_far = true_rows[1:t+1]
            for p in particles
                function ll_fn(rule::CARuleAST.RuleNode)
                    return log_likelihood_all(rule, rows_so_far, t; ε=EPSILON)
                end
                
                p.rule = CARejuvenation.rejuvenate_rule_with_likelihood(
                    p.rule, ll_fn;
                    num_steps=NUM_REJUV_STEPS,
                    max_depth=MAX_DEPTH
                )
                
                p.log_weight = log_likelihood_all(p.rule, rows_so_far, t; ε=EPSILON)
                p.log_weight += CAGrammarPrior.log_prior_tree(p.rule, P_LEAF)
            end
        end
    end
    
    return predictions
end

function load_trial_options()::Dict{Int, Dict{Int, Vector{Int}}}
    config_path = joinpath(@__DIR__, "trial_config.json")
    config = JSON.parsefile(config_path)
    
    options = Dict{Int, Dict{Int, Vector{Int}}}()
    
    for trial in config["trials"]
        rule_num = trial["rule_number"]
        time_step = trial["time_step"]
        
        if !haskey(options, rule_num)
            options[rule_num] = Dict{Int, Vector{Int}}()
        end
        
        candidate_rules = [opt["rule_applied"] for opt in trial["options"]]
        options[rule_num][time_step] = candidate_rules
    end
    
    return options
end

function main()
    Random.seed!(RANDOM_SEED)
    
    println("="^60)
    println("Generating Bayesian Model Predictions")
    println("="^60)
    println()
    
    println("Loading trial configuration...")
    trial_options = load_trial_options()
    
    all_predictions = Dict{String, Any}[]
    
    for rule_info in RULES_TO_TEST
        rule_num = rule_info.rule
        config_path = rule_info.config_path
        
        println("\nProcessing Rule $rule_num...")
        
        initial_row = load_initial_row(config_path)
        rule_options = trial_options[rule_num]
        
        predictions = generate_predictions_for_rule(rule_num, initial_row, rule_options)
        
        for t in TIME_STEPS_TO_TEST
            trial_id = @sprintf("rule_%03d_t%d", rule_num, t)
            pred = predictions[trial_id]
            
            candidate_probs = []
            for (i, rule) in enumerate(pred["candidate_rules"])
                push!(candidate_probs, Dict(
                    "option_id" => "opt_$(i-1)",
                    "rule" => rule,
                    "probability" => pred["candidate_probabilities"][i],
                    "is_correct" => (rule == rule_num)
                ))
            end
            
            push!(all_predictions, Dict(
                "trial_id" => trial_id,
                "rule_number" => rule_num,
                "time_step" => t,
                "candidate_probabilities" => candidate_probs
            ))
        end
    end
    
    sort!(all_predictions, by=x -> x["trial_id"])
    
    output = Dict(
        "model_config" => Dict(
            "num_particles" => NUM_PARTICLES,
            "epsilon" => EPSILON,
            "p_leaf" => P_LEAF,
            "max_depth" => MAX_DEPTH,
            "num_rejuv_steps" => NUM_REJUV_STEPS,
            "decision_temperature" => DECISION_TEMPERATURE,
            "ca_width" => CA_WIDTH,
            "model_type" => "resource_rational"
        ),
        "generated_at" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        "num_trials" => length(all_predictions),
        "trials" => all_predictions
    )
    
    output_path = joinpath(@__DIR__, "model_predictions.json")
    open(output_path, "w") do f
        JSON.print(f, output, 2)
    end
    
    println("\n" * "="^60)
    println("Model predictions saved to: $output_path")
    println("Total trials: $(length(all_predictions))")
    println("="^60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
