# Branching Impact Metrics in BaB Search

This document lists all readily available metrics to measure the "impact" of a branching decision in the Branch-and-Bound (BaB) search process.

## Overview

After a branching decision is made in `_parallel_dpll()` (verifier.py:484-500), the following information is available to measure impact:

1. **Before branching**: `pruned_ret` (AbstractResults) contains the state before splitting
2. **After branching**: `abstraction_ret` (AbstractResults) contains the state after abstraction
3. **After adding**: `self.domains_list` contains the updated state after pruning unverified branches

## Available Metrics

### 1. Problem Bounds Improvement

**Metric**: Change in `minimum_lowers` (worst-case lower bound across all domains)

**Location**: 
- Before: `self.domains_list.minimum_lowers` (before `add()`)
- After: `self.domains_list.minimum_lowers` (after `add()`)

**Code Reference**: 
- `verifier.py:504` - `minimum_lowers = self.domains_list.minimum_lowers`
- `domains_list.py:433-437` - Property definition

**Calculation**:
```python
bound_improvement = minimum_lowers_after - minimum_lowers_before
```

**Note**: Positive values indicate improvement (bounds got tighter/less negative).

---

### 2. Output Bounds Improvement

**Metric**: Change in output lower bounds (`output_lbs`) for each domain

**Location**:
- Before: `pruned_ret.output_lbs` (shape: `[batch, num_outputs]`)
- After: `abstraction_ret.output_lbs` (shape: `[2*batch, num_outputs]`)

**Code Reference**:
- `abstractor.py:435` - `'output_lbs': double_lower_bounds[self.net.final_name]`
- `domains_list.py:90, 218, 403` - Storage and retrieval

**Calculation**:
```python
# Compare parent domain with its two children
parent_lbs = pruned_ret.output_lbs[i]
child1_lbs = abstraction_ret.output_lbs[2*i]
child2_lbs = abstraction_ret.output_lbs[2*i + 1]

# Improvement = max of children - parent (for worst-case bound)
improvement = max(child1_lbs.max(), child2_lbs.max()) - parent_lbs.max()
```

---

### 3. Hidden Layer Bounds Improvement

**Metric**: Change in intermediate layer bounds (`lower_bounds`, `upper_bounds`)

**Location**:
- Before: `pruned_ret.lower_bounds`, `pruned_ret.upper_bounds` (dict per layer)
- After: `abstraction_ret.lower_bounds`, `abstraction_ret.upper_bounds` (dict per layer)

**Code Reference**:
- `abstractor.py:426` - `double_lower_bounds, double_upper_bounds = self.get_hidden_bounds(...)`
- `domains_list.py:117-118, 233-234, 391-393` - Storage

**Calculation**:
```python
# For each layer
for layer_name in pruned_ret.lower_bounds:
    parent_lb = pruned_ret.lower_bounds[layer_name][i]
    parent_ub = pruned_ret.upper_bounds[layer_name][i]
    
    child1_lb = abstraction_ret.lower_bounds[layer_name][2*i]
    child1_ub = abstraction_ret.upper_bounds[layer_name][2*i]
    child2_lb = abstraction_ret.lower_bounds[layer_name][2*i + 1]
    child2_ub = abstraction_ret.upper_bounds[layer_name][2*i + 1]
    
    # Bound width reduction
    parent_width = (parent_ub - parent_lb).sum()
    child_width = (child1_ub - child1_lb).sum() + (child2_ub - child2_lb).sum()
    width_reduction = parent_width - child_width
```

---

### 4. Phase-Fixed Neurons (New Stable Neurons)

**Metric**: Number of neurons that became phase-fixed (stable) after branching

**Location**:
- Before: `pruned_ret.masks` (dict: 1=active, -1=inactive, 0=unstable)
- After: `abstraction_ret.lower_bounds`, `abstraction_ret.upper_bounds` (can compute masks)

**Code Reference**:
- `util.py:21-30` - `compute_masks()` function
- `domains_list.py:256-260` - Mask computation
- `util.py:197-201` - Mask definition: `(lower > 0).int() - (upper < 0).int()`

**Calculation**:
```python
from heuristic.util import compute_masks

# Before branching
masks_before = pruned_ret.masks  # Already computed
unstable_before = {k: (v == 0).sum() for k, v in masks_before.items()}

# After branching
masks_after = compute_masks(
    lower_bounds=abstraction_ret.lower_bounds,
    upper_bounds=abstraction_ret.upper_bounds,
    device='cpu'
)
unstable_after = {k: (v == 0).sum() for k, v in masks_after.items()}

# Newly fixed neurons
newly_fixed = {}
for layer_name in unstable_before:
    # Each parent splits into 2 children
    parent_unstable = unstable_before[layer_name][i]
    child1_unstable = unstable_after[layer_name][2*i]
    child2_unstable = unstable_after[layer_name][2*i + 1]
    
    newly_fixed[layer_name] = parent_unstable - (child1_unstable + child2_unstable)
```

**Note**: This counts neurons that transitioned from unstable (0) to either active (1) or inactive (-1).

---

### 5. Domains Pruned (Verified)

**Metric**: Number of domains that were verified (pruned) after branching

**Location**:
- `domains_list.py:295` - `remaining_index` contains unverified domains
- `domains_list.py:369` - `unsat_indices` contains verified domains

**Code Reference**:
- `domains_list.py:289-423` - `add()` method
- `verifier.py:496-499` - Domain count tracking

**Calculation**:
```python
# In domains_list.add()
batch = len(domain_params.input_lowers)  # 2 * original batch
remaining_index = torch.where((domain_params.output_lbs <= domain_params.rhs).all(1))[0]
pruned_count = batch - len(remaining_index)

# Or track before/after
domains_before = len(self.domains_list)
self.domains_list.add(abstraction_ret, decisions)
domains_after = len(self.domains_list)
net_change = domains_after - domains_before  # Can be negative if many pruned
```

---

### 6. Unstable Neuron Count Change

**Metric**: Change in average number of unstable neurons per domain

**Location**:
- Before: `self.domains_list.count_unstable_neurons()` (before `add()`)
- After: `self.domains_list.count_unstable_neurons()` (after `add()`)

**Code Reference**:
- `domains_list.py:506-523` - `count_unstable_neurons()` method
- `verifier.py:446, 539-540` - Usage in logging

**Calculation**:
```python
unstable_before = self.domains_list.count_unstable_neurons()
self.domains_list.add(abstraction_ret, decisions)
unstable_after = self.domains_list.count_unstable_neurons()

# Per-domain change (accounts for domain count change)
if len(self.domains_list) > 0:
    avg_unstable_before = unstable_before
    avg_unstable_after = unstable_after
    change_per_domain = avg_unstable_after - avg_unstable_before
```

---

### 7. Boolean Constraint Propagation (BCP) Impact

**Metric**: Number of neurons fixed via BCP propagation (not explicit branching)

**Location**:
- `util.py:236-272` - `boolean_propagation()` function
- `domains_list.py:347-362` - BCP in `add()` method

**Code Reference**:
- `util.py:254` - `bcp_stat, bcp_vars = new_sat_solver.bcp()`
- `util.py:258-266` - Updates from BCP variables

**Calculation**:
```python
# In boolean_propagation() or add()
bcp_vars = []  # List of literals fixed by BCP
for idx in remaining_index:
    new_sat_solver = self.boolean_propagation(...)
    if new_sat_solver is not None:
        # bcp_vars contains newly fixed neurons
        bcp_vars.extend(bcp_vars_from_propagation)

# Count neurons fixed by BCP (not by explicit decision)
bcp_fixed_count = len(bcp_vars)
```

**Note**: This measures neurons fixed indirectly through constraint propagation, showing the "ripple effect" of the branching decision.

---

### 8. Input Bounds Tightening (Input Splitting Only)

**Metric**: Change in input domain size after input splitting

**Location**:
- Before: `pruned_ret.input_lowers`, `pruned_ret.input_uppers`
- After: `abstraction_ret.input_lowers`, `abstraction_ret.input_uppers`

**Code Reference**:
- `abstractor.py:459-463` - `input_split_idx()` for input splitting
- `domains_list.py:86-87, 211-212, 399-400` - Input bounds storage

**Calculation**:
```python
# Input domain volume
parent_volume = (pruned_ret.input_uppers[i] - pruned_ret.input_lowers[i]).prod()
child1_volume = (abstraction_ret.input_uppers[2*i] - abstraction_ret.input_lowers[2*i]).prod()
child2_volume = (abstraction_ret.input_uppers[2*i + 1] - abstraction_ret.input_lowers[2*i + 1]).prod()

# Volume reduction (should be ~50% for binary split)
volume_reduction = parent_volume - (child1_volume + child2_volume)
```

---

### 9. Decision History Depth

**Metric**: Increase in decision trail length (branching depth)

**Location**:
- Before: `pruned_ret.histories[i]` (dict of decision trails)
- After: `abstraction_ret.histories[2*i]`, `abstraction_ret.histories[2*i + 1]`

**Code Reference**:
- `abstractor.py:395` - `double_histories = self.update_histories(...)`
- `util.py:140-161` - `update_hidden_bounds_histories()` adds to history

**Calculation**:
```python
# Count decisions in history
def count_decisions(history):
    return sum(len(history[layer][0]) for layer in history)

parent_depth = count_decisions(pruned_ret.histories[i])
child1_depth = count_decisions(abstraction_ret.histories[2*i])
child2_depth = count_decisions(abstraction_ret.histories[2*i + 1])

# Depth increase
depth_increase = max(child1_depth, child2_depth) - parent_depth  # Should be 1
```

---

### 10. Domain Count Change

**Metric**: Net change in number of domains in the search tree

**Location**:
- `verifier.py:496-499` - Before/after domain counts

**Code Reference**:
- `verifier.py:496` - `domains_before = len(self.domains_list)`
- `verifier.py:498` - `self.domains_list.add(...)`
- `verifier.py:499` - `domains_after = len(self.domains_list)`

**Calculation**:
```python
domains_before = len(self.domains_list)
batch_picked = len(pick_ret.input_lowers)
self.domains_list.add(abstraction_ret, decisions)
domains_after = len(self.domains_list)

# Expected: domains_after = domains_before - batch_picked + remaining_after_split
# If many pruned: domains_after < domains_before - batch_picked + 2*batch_picked
net_domain_change = domains_after - domains_before
expected_change = 2 * batch_picked - batch_picked  # 2 children per parent
pruning_impact = expected_change - net_domain_change  # How many were pruned
```

---

### 11. Reference Bounds Improvement

**Metric**: Improvement relative to reference bounds (used in abstraction)

**Location**:
- `abstractor.py:403-404` - `reference_bounds` construction
- `abstractor.py:405-413` - Bounds computed with reference

**Code Reference**:
- `abstractor.py:403` - `double_ref_output_lbs = torch.cat([domain_params.output_lbs, domain_params.output_lbs], dim=0)`

**Calculation**:
```python
# Reference bounds are the parent's output bounds
reference_lbs = pruned_ret.output_lbs[i]
child1_lbs = abstraction_ret.output_lbs[2*i]
child2_lbs = abstraction_ret.output_lbs[2*i + 1]

# Improvement over reference
improvement1 = (child1_lbs - reference_lbs).max()
improvement2 = (child2_lbs - reference_lbs).max()
best_improvement = max(improvement1, improvement2)
```

---

### 12. Alpha/Slope Optimization Impact

**Metric**: Change in slope parameters (alpha) after branching

**Location**:
- Before: `pruned_ret.slopes` (dict per layer)
- After: `abstraction_ret.slopes` (dict per layer)

**Code Reference**:
- `abstractor.py:368-369, 422` - Slope handling
- `domains_list.py:97-104, 225-227, 410` - Slope storage

**Note**: Slopes are optimization parameters for bound tightening. Changes indicate how branching affects the optimization landscape.

---

## Implementation Example

Here's a code snippet showing how to compute multiple impact metrics:

```python
def compute_branching_impact(pruned_ret, abstraction_ret, domains_list_before, domains_list_after):
    """Compute comprehensive impact metrics for a branching decision."""
    
    metrics = {}
    
    # 1. Problem bounds improvement
    metrics['bound_improvement'] = (
        domains_list_after.minimum_lowers - domains_list_before.minimum_lowers
    )
    
    # 2. Output bounds improvement (per domain)
    batch = len(pruned_ret.output_lbs)
    output_improvements = []
    for i in range(batch):
        parent_lb = pruned_ret.output_lbs[i].max()
        child1_lb = abstraction_ret.output_lbs[2*i].max()
        child2_lb = abstraction_ret.output_lbs[2*i + 1].max()
        best_child = max(child1_lb, child2_lb)
        output_improvements.append(best_child - parent_lb)
    metrics['output_improvements'] = output_improvements
    metrics['avg_output_improvement'] = sum(output_improvements) / len(output_improvements)
    
    # 3. Phase-fixed neurons
    from heuristic.util import compute_masks
    masks_before = pruned_ret.masks
    masks_after = compute_masks(
        abstraction_ret.lower_bounds,
        abstraction_ret.upper_bounds,
        device='cpu'
    )
    
    newly_fixed = {}
    for layer_name in masks_before:
        unstable_before = (masks_before[layer_name] == 0).sum()
        unstable_after = (masks_after[layer_name] == 0).sum()
        newly_fixed[layer_name] = unstable_before - unstable_after
    metrics['newly_fixed_neurons'] = newly_fixed
    metrics['total_newly_fixed'] = sum(newly_fixed.values())
    
    # 4. Domain count change
    domains_before_count = len(domains_list_before)
    domains_after_count = len(domains_list_after)
    metrics['net_domain_change'] = domains_after_count - domains_before_count
    
    # 5. Unstable neuron count
    unstable_before = domains_list_before.count_unstable_neurons()
    unstable_after = domains_list_after.count_unstable_neurons()
    metrics['unstable_change'] = unstable_after - unstable_before
    
    return metrics
```

## Summary

The most readily available and meaningful metrics are:

1. **Problem bounds improvement** (`minimum_lowers` change) - Direct measure of verification progress
2. **Phase-fixed neurons** (unstable → stable transitions) - Shows constraint propagation impact
3. **Domains pruned** (verified count) - Shows how many subproblems were solved
4. **Output bounds improvement** - Per-domain bound tightening
5. **Unstable neuron reduction** - Search space reduction

These metrics can be computed with minimal overhead since all the required data is already available in the `AbstractResults` objects and `DomainsList` state.
