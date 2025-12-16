# MCMC rejuvenation moves for CA rule inference

module CARejuvenation

using Random
using ..CARuleAST
using ..CAGrammarPrior

export rejuvenate_rule
export grow_move, prune_move, swap_op_move, swap_atom_move, toggle_not_move


function grow_move(tree::RuleNode, max_depth::Int)
    atom_paths = get_atom_paths(tree)
    
    if count_nodes(tree) >= 2^max_depth - 1
        return nothing
    end
    
    path = rand(atom_paths)
    
    op = sample_random_op()
    left = sample_random_atom()
    right = sample_random_atom()
    new_subtree = BinaryOpNode(op, left, right)
    
    new_tree = replace_at_path(tree, path, new_subtree)
    
    n_atoms_old = length(atom_paths)
    n_binary_new = length(get_binary_paths(new_tree))
    log_ratio = log(n_binary_new) - log(n_atoms_old)
    
    return (new_tree, log_ratio)
end

function prune_move(tree::RuleNode)
    if tree isa AtomNode
        return nothing
    end
    
    binary_paths = get_binary_paths(tree)
    not_paths = get_not_paths(tree)
    all_complex_paths = vcat(binary_paths, not_paths)
    
    if isempty(all_complex_paths)
        return nothing
    end
    
    path = rand(all_complex_paths)
    new_atom = sample_random_atom()
    new_tree = replace_at_path(tree, path, new_atom)
    
    n_complex_old = length(all_complex_paths)
    n_atoms_new = length(get_atom_paths(new_tree))
    log_ratio = log(n_atoms_new) - log(n_complex_old)
    
    return (new_tree, log_ratio)
end

function swap_op_move(tree::RuleNode)
    binary_paths = get_binary_paths(tree)
    
    if isempty(binary_paths)
        return nothing
    end
    
    path = rand(binary_paths)
    old_node = get_node_at_path(tree, path)
    
    old_op = old_node.op
    other_ops = filter(op -> op != old_op, BINARY_OPS)
    new_op = rand(other_ops)
    
    new_node = BinaryOpNode(new_op, old_node.left, old_node.right)
    new_tree = replace_at_path(tree, path, new_node)
    
    return (new_tree, 0.0)
end

function swap_atom_move(tree::RuleNode)
    atom_paths = get_atom_paths(tree)
    
    if isempty(atom_paths)
        return nothing
    end
    
    path = rand(atom_paths)
    old_node = get_node_at_path(tree, path)
    
    old_atom = old_node.atom
    other_atoms = filter(a -> a != old_atom, ATOM_TYPES)
    new_atom = rand(other_atoms)
    
    new_node = AtomNode(new_atom)
    new_tree = replace_at_path(tree, path, new_node)
    
    return (new_tree, 0.0)
end

function toggle_not_move(tree::RuleNode, max_depth::Int)
    not_paths = get_not_paths(tree)
    can_remove = !isempty(not_paths)
    
    atom_paths = get_atom_paths(tree)
    binary_paths = get_binary_paths(tree)
    add_paths = vcat(atom_paths, binary_paths)
    can_add = !isempty(add_paths) && count_nodes(tree) < 2^max_depth - 1
    
    if !can_add && !can_remove
        return nothing
    end
    
    if can_add && can_remove
        add_not = rand() < 0.5
    elseif can_add
        add_not = true
    else
        add_not = false
    end
    
    if add_not
        path = rand(add_paths)
        old_node = get_node_at_path(tree, path)
        new_node = NotNode(old_node)
        new_tree = replace_at_path(tree, path, new_node)
        
        not_paths_new = get_not_paths(new_tree)
        log_ratio = log(length(not_paths_new)) - log(length(add_paths))
    else
        path = rand(not_paths)
        old_node = get_node_at_path(tree, path)
        new_tree = replace_at_path(tree, path, old_node.child)
        
        add_paths_new = vcat(get_atom_paths(new_tree), get_binary_paths(new_tree))
        log_ratio = log(length(add_paths_new)) - log(length(not_paths))
    end
    
    return (new_tree, log_ratio)
end


function structure_move(tree::RuleNode, max_depth::Int; 
                        p_grow::Float64=0.2,
                        p_prune::Float64=0.2,
                        p_swap_op::Float64=0.25,
                        p_swap_atom::Float64=0.25)
    u = rand()
    
    if u < p_grow
        result = grow_move(tree, max_depth)
    elseif u < p_grow + p_prune
        result = prune_move(tree)
    elseif u < p_grow + p_prune + p_swap_op
        result = swap_op_move(tree)
    elseif u < p_grow + p_prune + p_swap_op + p_swap_atom
        result = swap_atom_move(tree)
    else
        result = toggle_not_move(tree, max_depth)
    end
    
    if result === nothing
        result = swap_atom_move(tree)
    end
    
    if result === nothing
        return tree
    end
    
    new_tree, log_proposal_ratio = result
    
    size_old = count_nodes(tree)
    size_new = count_nodes(new_tree)
    p_leaf = CAGrammarPrior.p_leaf_default
    log_prior_ratio = (size_old - size_new) * log((1 - p_leaf) / p_leaf)
    
    log_accept = log_proposal_ratio + log_prior_ratio
    if log(rand()) < log_accept
        return new_tree
    else
        return tree
    end
end


function rejuvenate_rule(tree::RuleNode; num_steps::Int=3, max_depth::Int=5)
    current = tree
    for _ in 1:num_steps
        current = structure_move(current, max_depth)
    end
    return current
end

function rejuvenate_rule_with_likelihood(tree::RuleNode, log_likelihood_fn::Function;
                                         num_steps::Int=3, max_depth::Int=5)
    current = tree
    current_ll = log_likelihood_fn(current)
    p_leaf = CAGrammarPrior.p_leaf_default
    
    for _ in 1:num_steps
        u = rand()
        if u < 0.2
            result = grow_move(current, max_depth)
        elseif u < 0.4
            result = prune_move(current)
        elseif u < 0.65
            result = swap_op_move(current)
        elseif u < 0.9
            result = swap_atom_move(current)
        else
            result = toggle_not_move(current, max_depth)
        end
        
        if result === nothing
            continue
        end
        
        new_tree, log_proposal_ratio = result
        new_ll = log_likelihood_fn(new_tree)
        
        size_old = count_nodes(current)
        size_new = count_nodes(new_tree)
        log_prior_ratio = (size_old - size_new) * log((1 - p_leaf) / p_leaf)
        
        log_accept = (new_ll - current_ll) + log_proposal_ratio + log_prior_ratio
        
        if log(rand()) < log_accept
            current = new_tree
            current_ll = new_ll
        end
    end
    
    return current
end

export rejuvenate_rule_with_likelihood

end
