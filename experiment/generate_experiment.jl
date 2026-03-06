# Generate stimuli and trial configs for human CA rule inference experiment

using Plots
using JSON
using Random
using Dates
using Printf

const OUTPUT_DIR = joinpath(@__DIR__, "stimuli")
const WEBAPP_DIR = joinpath(@__DIR__, "webapp")
const NUM_OPTIONS = 4
const CA_WIDTH = 17
const TIME_STEPS_TO_TEST = 1:6
const RANDOM_SEED = 42

const RULES_TO_TEST = [
    (rule=32, class="I", description="Fixed - pattern dependent", config_path="rule_032/run_1_config.json"),
    (rule=5, class="II", description="Periodic structures", config_path="rule_005/run_1_config.json"),
    (rule=108, class="II", description="Periodic nested triangles", config_path="rule_108/run_1_config.json"),
    (rule=30, class="III", description="Chaotic - used for randomness", config_path="rule_030/run_1_config.json"),
    (rule=60, class="III", description="Chaotic XOR-based", config_path="rule_060/run_1_config.json"),
    (rule=54, class="IV", description="Complex localized structures", config_path="rule_054/run_3_config.json"),
    (rule=110, class="IV", description="Complex - Turing complete", config_path="rule_110/run_1_config.json"),
]

const CLASS_I_RULES = [0, 8, 32, 64, 128, 136, 160, 168, 192, 224, 234, 235, 238, 239, 248, 249, 250, 251, 252, 253, 254, 255]
const CLASS_II_RULES = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 19, 23, 24, 25, 27, 28, 29, 33, 34, 35, 36, 37, 38, 42, 43, 44, 46, 50, 51, 56, 57, 58, 62, 72, 73, 74, 76, 77, 78, 94, 104, 108, 130, 132, 138, 140, 142, 152, 154, 156, 162, 164, 170, 172, 178, 184, 200, 204, 232]
const CLASS_III_RULES = [18, 22, 26, 30, 45, 60, 73, 90, 105, 122, 126, 146, 150, 154, 182]
const CLASS_IV_RULES = [54, 106, 110]


function wolfram_rule_to_lookup_table(rule_number::Int)::Dict{Tuple{Bool,Bool,Bool}, Bool}
    @assert 0 <= rule_number <= 255 "Rule number must be 0-255"
    
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

function apply_rule(rule_number::Int, row::Vector{Bool})::Vector{Bool}
    lookup = wolfram_rule_to_lookup_table(rule_number)
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
        rows[t + 1] = apply_rule(rule_number, rows[t])
    end
    
    return rows
end


function create_ca_image(rows::Vector{Vector{Bool}}, 
                         total_height::Int,
                         filename::String;
                         highlight_next::Bool=false,
                         show_title::Bool=false,
                         title::String="")
    n_rows = length(rows)
    width = length(rows[1])
    
    grid = fill(0.5, total_height, width)
    
    for (t, row) in enumerate(rows)
        for (i, cell) in enumerate(row)
            grid[t, i] = cell ? 1.0 : 0.0
        end
    end
    
    plt = heatmap(grid,
                  color=cgrad([:white, :gray, :black]),
                  aspect_ratio=:equal,
                  yflip=true,
                  colorbar=false,
                  clims=(0, 1),
                  axis=false,
                  ticks=false,
                  border=:none,
                  framestyle=:none,
                  left_margin=-10Plots.mm,
                  right_margin=-10Plots.mm,
                  top_margin=-3Plots.mm,
                  bottom_margin=-3Plots.mm,
                  size=(width * 18, total_height * 18))
    
    grid_color = RGB(0.75, 0.75, 0.75)
    grid_linewidth = 0.5
    
    for x in 0.5:(width + 0.5)
        vline!([x], color=grid_color, linewidth=grid_linewidth, label=nothing)
    end
    
    for y in 0.5:(total_height + 0.5)
        hline!([y], color=grid_color, linewidth=grid_linewidth, label=nothing)
    end
    
    if highlight_next && n_rows < total_height
        hline!([n_rows + 0.5], 
               color=:red, 
               linewidth=3, 
               linestyle=:solid,
               label=nothing)
    end
    
    if show_title && !isempty(title)
        title!(title)
    end
    
    savefig(plt, filename)
    return filename
end

function create_question_image(rows::Vector{Vector{Bool}}, 
                                t::Int, 
                                total_height::Int,
                                filename::String)
    visible_rows = rows[1:t]
    create_ca_image(visible_rows, total_height, filename; highlight_next=true)
end

function create_answer_image(base_rows::Vector{Vector{Bool}}, 
                              t::Int,
                              answer_rule::Int,
                              total_height::Int,
                              filename::String)
    visible_rows = copy(base_rows[1:t])
    next_row = apply_rule(answer_rule, visible_rows[end])
    push!(visible_rows, next_row)
    create_ca_image(visible_rows, total_height, filename; highlight_next=false)
end


function get_same_class_rules(target_rule::Int, rule_class::String)::Vector{Int}
    if rule_class == "I"
        return filter(r -> r != target_rule, CLASS_I_RULES)
    elseif rule_class == "II"
        return filter(r -> r != target_rule, CLASS_II_RULES)
    elseif rule_class == "III"
        return filter(r -> r != target_rule, CLASS_III_RULES)
    elseif rule_class == "IV"
        return filter(r -> r != target_rule, CLASS_IV_RULES)
    else
        return Int[]
    end
end

function get_adjacent_rules(target_rule::Int)::Vector{Int}
    adjacent = Int[]
    if target_rule > 0
        push!(adjacent, target_rule - 1)
    end
    if target_rule < 255
        push!(adjacent, target_rule + 1)
    end
    return adjacent
end

function select_distractor_rules(target_rule::Int, rule_class::String)::Vector{Int}
    distractors = Int[]
    
    same_class = get_same_class_rules(target_rule, rule_class)
    adjacent = get_adjacent_rules(target_rule)
    
    if !isempty(adjacent)
        push!(distractors, shuffle(adjacent)[1])
    else
        available = setdiff(collect(0:255), [target_rule], distractors)
        push!(distractors, shuffle(available)[1])
    end
    
    available_same_class = setdiff(same_class, distractors)
    if !isempty(available_same_class)
        push!(distractors, shuffle(collect(available_same_class))[1])
    else
        available = setdiff(collect(0:255), [target_rule], distractors)
        push!(distractors, shuffle(available)[1])
    end
    
    available = setdiff(collect(0:255), [target_rule], distractors)
    push!(distractors, shuffle(available)[1])
    
    return distractors[1:3]
end


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

function generate_all_trials()
    Random.seed!(RANDOM_SEED)
    
    mkpath(OUTPUT_DIR)
    mkpath(WEBAPP_DIR)
    
    all_trials = []
    rule_metadata = Dict()
    
    total_height = maximum(TIME_STEPS_TO_TEST) + 2
    
    println("Generating experiment stimuli...")
    println("="^50)
    
    for rule_info in RULES_TO_TEST
        rule_num = rule_info.rule
        rule_class = rule_info.class
        rule_desc = rule_info.description
        config_path = rule_info.config_path
        
        println("\nProcessing Rule $(rule_num) (Class $(rule_class))...")
        
        rule_metadata[string(rule_num)] = Dict(
            "class" => rule_class,
            "description" => rule_desc
        )
        
        rule_dir = joinpath(OUTPUT_DIR, @sprintf("rule_%03d", rule_num))
        mkpath(rule_dir)
        
        initial_row = load_initial_row(config_path)
        
        max_t = maximum(TIME_STEPS_TO_TEST) + 1
        true_evolution = generate_ca_evolution(rule_num, initial_row, max_t)
        
        for t in TIME_STEPS_TO_TEST
            println("  t=$(t)...")
            
            trial_id = "rule_$(lpad(rule_num, 3, '0'))_t$(t)"
            
            question_file = joinpath(rule_dir, "question_t$(t).png")
            create_question_image(true_evolution, t, total_height, question_file)
            
            distractor_rules = select_distractor_rules(rule_num, rule_class)
            all_option_rules = vcat([rule_num], distractor_rules)
            
            options = []
            for (opt_idx, opt_rule) in enumerate(all_option_rules)
                opt_id = "opt_$(opt_idx - 1)"
                answer_file = joinpath(rule_dir, "answer_t$(t)_$(opt_id)_rule$(opt_rule).png")
                
                create_answer_image(true_evolution, t, opt_rule, total_height, answer_file)
                
                rel_path = joinpath("stimuli", @sprintf("rule_%03d", rule_num), basename(answer_file))
                
                push!(options, Dict(
                    "id" => opt_id,
                    "image" => rel_path,
                    "rule_applied" => opt_rule,
                    "is_correct" => (opt_rule == rule_num)
                ))
            end
            
            option_order = shuffle(0:(NUM_OPTIONS-1))
            
            trial = Dict(
                "trial_id" => trial_id,
                "rule_number" => rule_num,
                "rule_class" => "Class $(rule_class) - $(rule_desc)",
                "time_step" => t,
                "question_image" => joinpath("stimuli", @sprintf("rule_%03d", rule_num), "question_t$(t).png"),
                "options" => options,
                "option_order" => option_order
            )
            
            push!(all_trials, trial)
        end
    end
    
    rule_groups = Dict()
    for trial in all_trials
        rule = trial["rule_number"]
        if !haskey(rule_groups, rule)
            rule_groups[rule] = []
        end
        push!(rule_groups[rule], trial)
    end
    
    shuffled_rules = shuffle(collect(keys(rule_groups)))
    shuffled_trials = []
    for rule in shuffled_rules
        sorted_trials = sort(rule_groups[rule], by=t -> t["time_step"])
        append!(shuffled_trials, sorted_trials)
    end
    
    experiment_config = Dict(
        "experiment_id" => "ca_rule_inference_v1",
        "generated_at" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        "num_trials" => length(shuffled_trials),
        "num_rules" => length(RULES_TO_TEST),
        "time_steps" => collect(TIME_STEPS_TO_TEST),
        "trials" => shuffled_trials,
        "rule_metadata" => rule_metadata
    )
    
    config_file = joinpath(@__DIR__, "trial_config.json")
    open(config_file, "w") do f
        JSON.print(f, experiment_config, 2)
    end
    
    println("\n" * "="^50)
    println("Generation complete!")
    println("  Total trials: $(length(shuffled_trials))")
    println("  Config file: $(config_file)")
    println("  Stimuli directory: $(OUTPUT_DIR)")
end


function main()
    generate_all_trials()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
