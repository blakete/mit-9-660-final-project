# Probabilistic grammar for CA rule trees using Gen

module CAGrammarPrior

using Gen
using ..CARuleAST

export sample_rule_tree, p_leaf_default
export ATOM_TYPES, BINARY_OPS

const p_leaf_default = 0.5
const p_not = 0.2
const ATOM_TYPES = [:left, :center, :right, :one, :zero]
const BINARY_OPS = [:and, :or, :xor]


@gen function sample_atom()
    weights = [0.25, 0.25, 0.25, 0.125, 0.125]
    atom_idx ~ categorical(weights)
    atom = ATOM_TYPES[atom_idx]
    return AtomNode(atom)
end

@gen function sample_binary_op()
    op_idx ~ categorical([1/3, 1/3, 1/3])
    return BINARY_OPS[op_idx]
end

@gen function sample_rule_tree_recursive(depth::Int, p_leaf::Float64, max_depth::Int)
    if depth >= max_depth
        node = {:atom} ~ sample_atom()
        return node
    end
    
    is_leaf ~ bernoulli(p_leaf)
    
    if is_leaf
        node = {:atom} ~ sample_atom()
        return node
    else
        is_not ~ bernoulli(p_not)
        
        if is_not
            child = {:child} ~ sample_rule_tree_recursive(depth + 1, p_leaf, max_depth)
            return NotNode(child)
        else
            op ~ categorical([1/3, 1/3, 1/3])
            op_sym = BINARY_OPS[op]
            
            left = {:left} ~ sample_rule_tree_recursive(depth + 1, p_leaf, max_depth)
            right = {:right} ~ sample_rule_tree_recursive(depth + 1, p_leaf, max_depth)
            
            return BinaryOpNode(op_sym, left, right)
        end
    end
end

@gen function sample_rule_tree(p_leaf::Float64, max_depth::Int)
    tree = {:tree} ~ sample_rule_tree_recursive(0, p_leaf, max_depth)
    return tree
end


function log_prior_tree(tree::RuleNode, p_leaf::Float64=p_leaf_default)::Float64
    n_nodes = count_nodes(tree)
    return (n_nodes - 1) * log(1 - p_leaf)
end

export log_prior_tree


function sample_random_atom()::AtomNode
    weights = [0.25, 0.25, 0.25, 0.125, 0.125]
    cumweights = cumsum(weights)
    u = rand()
    idx = searchsortedfirst(cumweights, u)
    idx = clamp(idx, 1, length(ATOM_TYPES))
    return AtomNode(ATOM_TYPES[idx])
end

function sample_random_op()::Symbol
    return rand(BINARY_OPS)
end

function sample_random_subtree(max_depth::Int=2)::RuleNode
    if max_depth <= 0 || rand() < p_leaf_default
        return sample_random_atom()
    end
    
    if rand() < p_not
        child = sample_random_subtree(max_depth - 1)
        return NotNode(child)
    else
        op = sample_random_op()
        left = sample_random_subtree(max_depth - 1)
        right = sample_random_subtree(max_depth - 1)
        return BinaryOpNode(op, left, right)
    end
end

export sample_random_atom, sample_random_op, sample_random_subtree

end
