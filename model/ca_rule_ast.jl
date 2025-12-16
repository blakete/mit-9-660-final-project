# AST representation for 1D CA rules

module CARuleAST

using Printf

export RuleNode, AtomNode, NotNode, BinaryOpNode
export eval_rule, count_nodes, count_atoms
export rule_to_string, rule_to_expr_string
export get_all_atoms, get_atom_paths, get_binary_paths, get_not_paths
export get_node_at_path, replace_at_path
export rule_to_lookup_table, lookup_table_to_rule_number
export wolfram_rule_to_lookup_table, compare_rules

abstract type RuleNode end

struct AtomNode <: RuleNode
    atom::Symbol
end

const VALID_ATOMS = [:left, :center, :right, :one, :zero]

struct NotNode <: RuleNode
    child::RuleNode
end

struct BinaryOpNode <: RuleNode
    op::Symbol
    left::RuleNode
    right::RuleNode
end

const VALID_OPS = [:and, :or, :xor]


function eval_atom(atom::Symbol, left::Bool, center::Bool, right::Bool)::Bool
    if atom == :left
        return left
    elseif atom == :center
        return center
    elseif atom == :right
        return right
    elseif atom == :one
        return true
    else
        return false
    end
end

function apply_op(op::Symbol, a::Bool, b::Bool)::Bool
    if op == :and
        return a && b
    elseif op == :or
        return a || b
    else
        return xor(a, b)
    end
end

function eval_rule(node::AtomNode, left::Bool, center::Bool, right::Bool)::Bool
    return eval_atom(node.atom, left, center, right)
end

function eval_rule(node::NotNode, left::Bool, center::Bool, right::Bool)::Bool
    return !eval_rule(node.child, left, center, right)
end

function eval_rule(node::BinaryOpNode, left::Bool, center::Bool, right::Bool)::Bool
    left_val = eval_rule(node.left, left, center, right)
    right_val = eval_rule(node.right, left, center, right)
    return apply_op(node.op, left_val, right_val)
end


function count_nodes(node::AtomNode)::Int
    return 1
end

function count_nodes(node::NotNode)::Int
    return 1 + count_nodes(node.child)
end

function count_nodes(node::BinaryOpNode)::Int
    return 1 + count_nodes(node.left) + count_nodes(node.right)
end

function count_atoms(node::AtomNode)::Int
    return 1
end

function count_atoms(node::NotNode)::Int
    return count_atoms(node.child)
end

function count_atoms(node::BinaryOpNode)::Int
    return count_atoms(node.left) + count_atoms(node.right)
end

function get_all_atoms(node::AtomNode)::Vector{Symbol}
    return [node.atom]
end

function get_all_atoms(node::NotNode)::Vector{Symbol}
    return get_all_atoms(node.child)
end

function get_all_atoms(node::BinaryOpNode)::Vector{Symbol}
    return vcat(get_all_atoms(node.left), get_all_atoms(node.right))
end


function get_atom_paths(node::AtomNode, path::Vector{Symbol}=Symbol[])
    return [copy(path)]
end

function get_atom_paths(node::NotNode, path::Vector{Symbol}=Symbol[])
    return get_atom_paths(node.child, vcat(path, [:child]))
end

function get_atom_paths(node::BinaryOpNode, path::Vector{Symbol}=Symbol[])
    left_paths = get_atom_paths(node.left, vcat(path, [:left]))
    right_paths = get_atom_paths(node.right, vcat(path, [:right]))
    return vcat(left_paths, right_paths)
end

function get_binary_paths(node::AtomNode, path::Vector{Symbol}=Symbol[])
    return Vector{Vector{Symbol}}()
end

function get_binary_paths(node::NotNode, path::Vector{Symbol}=Symbol[])
    return get_binary_paths(node.child, vcat(path, [:child]))
end

function get_binary_paths(node::BinaryOpNode, path::Vector{Symbol}=Symbol[])
    current = [copy(path)]
    left_paths = get_binary_paths(node.left, vcat(path, [:left]))
    right_paths = get_binary_paths(node.right, vcat(path, [:right]))
    return vcat(current, left_paths, right_paths)
end

function get_not_paths(node::AtomNode, path::Vector{Symbol}=Symbol[])
    return Vector{Vector{Symbol}}()
end

function get_not_paths(node::NotNode, path::Vector{Symbol}=Symbol[])
    current = [copy(path)]
    child_paths = get_not_paths(node.child, vcat(path, [:child]))
    return vcat(current, child_paths)
end

function get_not_paths(node::BinaryOpNode, path::Vector{Symbol}=Symbol[])
    left_paths = get_not_paths(node.left, vcat(path, [:left]))
    right_paths = get_not_paths(node.right, vcat(path, [:right]))
    return vcat(left_paths, right_paths)
end

function get_node_at_path(node::RuleNode, path::Vector{Symbol})
    if isempty(path)
        return node
    end
    
    direction = path[1]
    rest = path[2:end]
    
    if node isa AtomNode
        error("Cannot traverse into atom node")
    elseif node isa NotNode
        if direction != :child
            error("Invalid direction for NotNode: $direction")
        end
        return get_node_at_path(node.child, rest)
    else
        if direction == :left
            return get_node_at_path(node.left, rest)
        elseif direction == :right
            return get_node_at_path(node.right, rest)
        else
            error("Invalid direction for BinaryOpNode: $direction")
        end
    end
end

"Replace node at path, returning new tree."
function replace_at_path(node::RuleNode, path::Vector{Symbol}, new_node::RuleNode)
    if isempty(path)
        return new_node
    end
    
    direction = path[1]
    rest = path[2:end]
    
    if node isa AtomNode
        error("Cannot traverse into atom node")
    elseif node isa NotNode
        if direction != :child
            error("Invalid direction for NotNode: $direction")
        end
        new_child = replace_at_path(node.child, rest, new_node)
        return NotNode(new_child)
    else
        if direction == :left
            new_left = replace_at_path(node.left, rest, new_node)
            return BinaryOpNode(node.op, new_left, node.right)
        elseif direction == :right
            new_right = replace_at_path(node.right, rest, new_node)
            return BinaryOpNode(node.op, node.left, new_right)
        else
            error("Invalid direction for BinaryOpNode: $direction")
        end
    end
end


function atom_to_string(atom::Symbol)::String
    if atom == :left
        return "L"
    elseif atom == :center
        return "C"
    elseif atom == :right
        return "R"
    elseif atom == :one
        return "1"
    else
        return "0"
    end
end

function op_to_string(op::Symbol)::String
    if op == :and
        return "∧"
    elseif op == :or
        return "∨"
    else
        return "⊕"
    end
end

function rule_to_string(node::AtomNode)::String
    return atom_to_string(node.atom)
end

function rule_to_string(node::NotNode)::String
    return "¬" * rule_to_string(node.child)
end

function rule_to_string(node::BinaryOpNode)::String
    left_str = rule_to_string(node.left)
    right_str = rule_to_string(node.right)
    op_str = op_to_string(node.op)
    return "($left_str $op_str $right_str)"
end

function rule_to_expr_string(node::AtomNode)::String
    if node.atom == :left
        return "left"
    elseif node.atom == :center
        return "center"
    elseif node.atom == :right
        return "right"
    elseif node.atom == :one
        return "true"
    else
        return "false"
    end
end

function rule_to_expr_string(node::NotNode)::String
    return "NOT(" * rule_to_expr_string(node.child) * ")"
end

function rule_to_expr_string(node::BinaryOpNode)::String
    left_str = rule_to_expr_string(node.left)
    right_str = rule_to_expr_string(node.right)
    op_name = uppercase(String(node.op))
    return "$op_name($left_str, $right_str)"
end


function rule_to_lookup_table(rule::RuleNode)::Dict{Tuple{Bool,Bool,Bool}, Bool}
    table = Dict{Tuple{Bool,Bool,Bool}, Bool}()
    for l in [false, true]
        for c in [false, true]
            for r in [false, true]
                table[(l, c, r)] = eval_rule(rule, l, c, r)
            end
        end
    end
    return table
end

"Convert lookup table to Wolfram rule number (0-255)."
function lookup_table_to_rule_number(table::Dict{Tuple{Bool,Bool,Bool}, Bool})::Int
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
    
    rule_number = 0
    for (i, neigh) in enumerate(neighborhoods)
        if table[neigh]
            rule_number += 2^(8 - i)
        end
    end
    return rule_number
end

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

function rule_to_wolfram_number(rule::RuleNode)::Int
    table = rule_to_lookup_table(rule)
    return lookup_table_to_rule_number(table)
end

function compare_rules(rule1::RuleNode, rule2::RuleNode)::Bool
    for l in [false, true]
        for c in [false, true]
            for r in [false, true]
                if eval_rule(rule1, l, c, r) != eval_rule(rule2, l, c, r)
                    return false
                end
            end
        end
    end
    return true
end

function rule_hamming_distance(rule1::RuleNode, rule2::RuleNode)::Int
    distance = 0
    for l in [false, true]
        for c in [false, true]
            for r in [false, true]
                if eval_rule(rule1, l, c, r) != eval_rule(rule2, l, c, r)
                    distance += 1
                end
            end
        end
    end
    return distance
end


function make_rule_90()::RuleNode
    return BinaryOpNode(:xor, AtomNode(:left), AtomNode(:right))
end

function make_rule_30()::RuleNode
    return BinaryOpNode(:xor, 
                        AtomNode(:left), 
                        BinaryOpNode(:or, AtomNode(:center), AtomNode(:right)))
end

function make_rule_110()::RuleNode
    center_xor_right = BinaryOpNode(:xor, AtomNode(:center), AtomNode(:right))
    not_left = NotNode(AtomNode(:left))
    center_and_right = BinaryOpNode(:and, AtomNode(:center), AtomNode(:right))
    three_way = BinaryOpNode(:and, not_left, center_and_right)
    return BinaryOpNode(:or, center_xor_right, three_way)
end

export make_rule_90, make_rule_30, make_rule_110
export rule_to_wolfram_number, rule_hamming_distance

end
