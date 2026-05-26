"""Knapsack cuts: topology-aware, prune-only cuts inferred at verified leaves.

Ported from alpha-beta-CROWN `complete_verifier/cuts/knapsack_cuts.py`.
See HANDOFF.md in the Verifier_Development repo for the full design write-up.

For each "furthest" fixed ReLU target b at a verified BaB leaf, pre_b is
expressed as a linear function of upstream RELU f-vars and INPUT vars:

    pre_b = constant + sum_j w_j * x_j     (x_j upstream RELU f-var or INPUT)

At the verified leaf the leaf forced pre_b into one half-line:
    ACTIVE  : pre_b >= threshold   (threshold = leaf's lb on pre_b)
    INACTIVE: pre_b <= threshold   (threshold = leaf's ub on pre_b)

The cut stores ONLY static topology data (weights, constant, threshold).
Bounds are queried fresh from the current subdomain at every check.
"""

from typing import Optional

import torch

from abstractor.auto_LiRPA.operators import (
    BoundAdd,
    BoundBatchNormalization,
    BoundBuffers,
    BoundConv,
    BoundFlatten,
    BoundInput,
    BoundLinear,
    BoundParams,
    BoundRelu,
    BoundReshape,
    BoundSqueeze,
    BoundSub,
    BoundUnsqueeze,
)

from setting import Settings


class KnapsackCut:
    """One cut: pre_b expressed as linear form over upstream f-vars / inputs."""

    __slots__ = (
        "target_b_var",
        "is_active",
        "relu_layer_idx",
        "neuron_idx",
        "coefficients",
        "constant",
        "threshold",
    )

    def __init__(self):
        self.target_b_var: Optional[str] = None
        self.is_active: bool = False
        self.relu_layer_idx: int = -1
        self.neuron_idx: int = -1
        # var_key -> coefficient. var_key is (layer_idx, flat_neuron_idx) for
        # upstream f-vars, or ("input", flat_idx) for input vars.
        self.coefficients: dict = {}
        self.constant: float = 0.0
        self.threshold: float = 0.0


class KnapsackCutGroup:
    """All cuts harvested from one verified leaf. Group implied iff every cut implied."""

    __slots__ = ("cuts", "depth")

    def __init__(self):
        self.cuts: list = []
        self.depth: int = 0


class KnapsackCutManager:
    """Owns lifecycle of knapsack cut groups.

    Precomputes downstream-ReLU reachability at solve start, collects a cut
    group at each verified leaf, checks at every new subproblem whether any
    stored group is implied.
    """

    def __init__(self):
        self._initialized = False
        self._debug = False
        self._net = None  # BoundedModule
        self._cut_groups: list = []
        self._num_cuts_built = 0
        self._num_prune_checks_rows = 0
        self._num_prunes_rows = 0

        # Pre-activation layer name -> int index.
        self._key_mapping: dict = {}
        self._key_mapping_inv: dict = {}

        # ReLU op handles, in net.relus topo order.
        self._relus: list = []

        # Layer-DAG forward reachability:
        #   _downstream_relus[layer_idx] = set of downstream layer indices
        self._downstream_relus: dict = {}

        # Convenience: pre-activation layer name -> ReLU op.
        self._preact_name_to_relu: dict = {}
        # Layer index -> ReLU op.
        self._layer_idx_to_relu: dict = {}

    def initialize(self, bound_module, ret_lower_bounds=None):
        """One-time setup. Builds layer-DAG forward reachability via BFS.

        bound_module: auto_LiRPA BoundedModule (NeuralSAT: abstractor.net).
        ret_lower_bounds: optional dict (e.g. AbstractResults.lower_bounds)
            whose iteration order determines key_mapping. Without it falls
            back to net.relus order + final layer.
        """
        self._net = bound_module
        self._cut_groups = []
        self._num_cuts_built = 0
        self._num_prune_checks_rows = 0
        self._num_prunes_rows = 0
        self._debug = bool(getattr(Settings, "use_knapsack_cuts_debug", False))
        self._relus = list(bound_module.relus)
        self._preact_name_to_relu = {r.inputs[0].name: r for r in self._relus}

        # Build key_mapping. Prefer ret iteration order.
        if ret_lower_bounds is not None:
            self._key_mapping = {key: i for i, key in enumerate(ret_lower_bounds.keys())}
        else:
            final_name = getattr(bound_module, "final_node_name", None) or getattr(
                bound_module, "final_name", None
            )
            keys = [r.inputs[0].name for r in self._relus]
            if final_name is not None and final_name not in keys:
                keys.append(final_name)
            self._key_mapping = {k: i for i, k in enumerate(keys)}
        self._key_mapping_inv = {i: k for k, i in self._key_mapping.items()}
        self._layer_idx_to_relu = {
            self._key_mapping[r.inputs[0].name]: r for r in self._relus
        }

        # Layer-DAG BFS: for each ReLU, BFS forward through node.output_name
        # collecting any ReLU encountered.
        relu_names = {r.name for r in self._relus}

        for r in self._relus:
            my_idx = self._key_mapping[r.inputs[0].name]
            downstream: set = set()
            visited = {r.name}
            queue = [r]
            while queue:
                cur = queue.pop()
                for succ_name in cur.output_name:
                    if succ_name in visited:
                        continue
                    visited.add(succ_name)
                    succ = bound_module[succ_name]
                    if succ_name in relu_names:
                        succ_idx = self._key_mapping[succ.inputs[0].name]
                        downstream.add(succ_idx)
                    queue.append(succ)
            self._downstream_relus[my_idx] = downstream

        self._initialized = True

        total_edges = sum(len(s) for s in self._downstream_relus.values())
        max_ds = max((len(s) for s in self._downstream_relus.values()), default=0)
        print(
            f"[knapsack] initialize: tracked_relus={len(self._relus)} "
            f"total_downstream_edges={total_edges} "
            f"max_downstream_per_relu={max_ds}"
        )

    def build_cut(self, relu_op, neuron_idx, active, threshold):
        if not self._initialized:
            return None
        preact = relu_op.inputs[0]

        cut = KnapsackCut()
        cut.target_b_var = preact.name
        cut.is_active = active
        cut.relu_layer_idx = self._key_mapping[preact.name]
        cut.neuron_idx = neuron_idx
        cut.coefficients = {}
        cut.constant = 0.0
        cut.threshold = float(threshold)

        if not self._fold_backward(preact, neuron_idx, 1.0, cut, depth=0):
            return None
        return cut

    # ---------- fold_backward ----------

    _MAX_FOLD_DEPTH = 64

    def _fold_backward(self, node, neuron_idx, weight, cut, depth):
        if depth > self._MAX_FOLD_DEPTH:
            print(f"[knapsack] fold_backward: max depth {self._MAX_FOLD_DEPTH} exceeded")
            return False

        if isinstance(node, BoundRelu):
            layer_idx = self._key_mapping[node.inputs[0].name]
            key = (layer_idx, int(neuron_idx))
            cut.coefficients[key] = cut.coefficients.get(key, 0.0) + weight
            return True

        if isinstance(node, BoundInput) and not isinstance(node, (BoundParams, BoundBuffers)):
            key = ("input", int(neuron_idx))
            cut.coefficients[key] = cut.coefficients.get(key, 0.0) + weight
            return True

        if isinstance(node, (BoundParams, BoundBuffers)):
            print(f"[knapsack] fold_backward: hit BoundParams/Buffers {node.name} unexpectedly")
            return False

        if isinstance(node, BoundLinear):
            return self._fold_linear(node, neuron_idx, weight, cut, depth)

        if isinstance(node, BoundConv):
            return self._fold_conv(node, neuron_idx, weight, cut, depth)

        if isinstance(node, (BoundReshape, BoundFlatten, BoundUnsqueeze, BoundSqueeze)):
            return self._fold_backward(node.inputs[0], neuron_idx, weight, cut, depth + 1)

        if isinstance(node, BoundAdd):
            data_inputs = [
                inp for inp in node.inputs
                if not isinstance(inp, (BoundParams, BoundBuffers))
            ]
            for inp in data_inputs:
                if not self._fold_backward(inp, neuron_idx, weight, cut, depth + 1):
                    return False
            return True

        if isinstance(node, BoundSub):
            data_inputs = [
                inp for inp in node.inputs
                if not isinstance(inp, (BoundParams, BoundBuffers))
            ]
            if len(data_inputs) != 2:
                print(f"[knapsack] fold_backward: BoundSub expects 2 data inputs, got {len(data_inputs)}")
                return False
            if not self._fold_backward(data_inputs[0], neuron_idx, weight, cut, depth + 1):
                return False
            if not self._fold_backward(data_inputs[1], neuron_idx, -weight, cut, depth + 1):
                return False
            return True

        if isinstance(node, BoundBatchNormalization):
            return self._fold_bn(node, neuron_idx, weight, cut, depth)

        print(f"[knapsack] fold_backward: unsupported op {type(node).__name__} ({node.name})")
        return False

    def _fold_linear(self, node, neuron_idx, weight, cut, depth):
        data_input = node.inputs[0]
        w_node = node.inputs[1]
        W = w_node.param if hasattr(w_node, "param") else w_node.value
        has_bias = len(node.inputs) > 2

        in_dim = self._flat_size(data_input)
        out_dim = self._flat_size(node)
        if in_dim is None or out_dim is None:
            print(f"[knapsack] fold_linear: shape unavailable on {node.name}")
            return False

        if W.shape == (out_dim, in_dim):
            row = W[neuron_idx, :]
        elif W.shape == (in_dim, out_dim):
            row = W[:, neuron_idx]
        else:
            print(f"[knapsack] fold_linear: weight shape {tuple(W.shape)} doesn't match in={in_dim} out={out_dim}")
            return False

        if has_bias:
            b_node = node.inputs[2]
            B = b_node.param if hasattr(b_node, "param") else b_node.value
            cut.constant += weight * float(B.flatten()[neuron_idx].item())

        nonzero = torch.nonzero(row, as_tuple=True)[0].tolist()
        for in_idx in nonzero:
            w_prime = float(row[in_idx].item())
            if not self._fold_backward(data_input, in_idx, weight * w_prime, cut, depth + 1):
                return False
        return True

    def _fold_conv(self, node, neuron_idx, weight, cut, depth):
        if getattr(node, "conv_dim", 2) != 2:
            print(f"[knapsack] fold_conv: only 2D conv supported (got conv_dim={node.conv_dim})")
            return False

        data_input = node.inputs[0]
        w_node = node.inputs[1]
        W = w_node.param if hasattr(w_node, "param") else w_node.value
        has_bias = getattr(node, "has_bias", False) and len(node.inputs) > 2

        in_shape = self._spatial_shape(data_input)
        out_shape = self._spatial_shape(node)
        if in_shape is None or out_shape is None:
            print(f"[knapsack] fold_conv: shape unavailable on {node.name}")
            return False
        c_in, h_in, w_in = in_shape
        c_out, h_out, w_out = out_shape

        weight_c_out, weight_c_in_per_group, k_h, k_w = W.shape
        groups = node.groups
        c_in_per_group = c_in // groups
        c_out_per_group = c_out // groups

        c = neuron_idx // (h_out * w_out)
        rem = neuron_idx % (h_out * w_out)
        h_o = rem // w_out
        w_o = rem % w_out
        group_id = c // c_out_per_group

        if has_bias:
            b_node = node.inputs[2]
            B = b_node.param if hasattr(b_node, "param") else b_node.value
            cut.constant += weight * float(B.flatten()[c].item())

        stride_h = node.stride[0]
        stride_w = node.stride[1] if len(node.stride) > 1 else node.stride[0]
        pad_h = node.padding[0]
        pad_w = node.padding[1] if len(node.padding) > 1 else node.padding[0]

        for kh in range(k_h):
            h_i = h_o * stride_h + kh - pad_h
            if h_i < 0 or h_i >= h_in:
                continue
            for kw in range(k_w):
                w_i = w_o * stride_w + kw - pad_w
                if w_i < 0 or w_i >= w_in:
                    continue
                slice_kc = W[c, :, kh, kw]
                nonzero_kc = torch.nonzero(slice_kc, as_tuple=True)[0].tolist()
                for k_c_local in nonzero_kc:
                    c_i = group_id * c_in_per_group + k_c_local
                    w_val = float(slice_kc[k_c_local].item())
                    flat_in = c_i * (h_in * w_in) + h_i * w_in + w_i
                    if not self._fold_backward(data_input, flat_in, weight * w_val, cut, depth + 1):
                        return False
        return True

    def _fold_bn(self, node, neuron_idx, weight, cut, depth):
        data_input = node.inputs[0]
        shape = self._spatial_shape(node)
        if shape is None:
            print(f"[knapsack] fold_bn: shape unavailable on {node.name}")
            return False
        c_total, h, w = shape
        c = neuron_idx // (h * w)
        gamma = node.inputs[1].param if hasattr(node.inputs[1], "param") else node.inputs[1].value
        beta_ = node.inputs[2].param if hasattr(node.inputs[2], "param") else node.inputs[2].value
        mean = (
            node.inputs[3].param if hasattr(node.inputs[3], "param")
            else node.inputs[3].buffer if hasattr(node.inputs[3], "buffer")
            else node.inputs[3].value
        )
        var = (
            node.inputs[4].param if hasattr(node.inputs[4], "param")
            else node.inputs[4].buffer if hasattr(node.inputs[4], "buffer")
            else node.inputs[4].value
        )
        eps = float(getattr(node, "eps", 1e-5))
        s = float((gamma[c] / torch.sqrt(var[c] + eps)).item())
        sh = float((beta_[c] - mean[c] * gamma[c] / torch.sqrt(var[c] + eps)).item())
        cut.constant += weight * sh
        return self._fold_backward(data_input, neuron_idx, weight * s, cut, depth + 1)

    # ---------- shape helpers ----------

    @staticmethod
    def _flat_size(node):
        shp = getattr(node, "output_shape", None)
        if shp is not None:
            return int(torch.prod(torch.tensor(list(shp)[1:])).item())
        fn = getattr(node, "flattened_nodes", None)
        if fn:
            return int(fn)
        return None

    @staticmethod
    def _spatial_shape(node):
        shp = getattr(node, "output_shape", None)
        if shp is None or len(shp) != 4:
            return None
        return int(shp[1]), int(shp[2]), int(shp[3])

    def reset(self):
        self._cut_groups = []
        self._num_cuts_built = 0
        self._num_prune_checks_rows = 0
        self._num_prunes_rows = 0
        self._initialized = False

    # ---------- history-format helpers ----------

    @staticmethod
    def _hist_to_tensors(loc, sign):
        """NeuralSAT histories store (loc, sign, beta) as either tensors or
        empty Python lists (depth-0 init). Normalize to tensors."""
        if isinstance(loc, torch.Tensor):
            return loc, sign
        if len(loc) == 0:
            return None, None
        return (
            torch.as_tensor(loc, dtype=torch.long),
            torch.as_tensor(sign),
        )

    # ---------- collection ----------

    def collect_from_verified(self, histories, output_lbs, rhs,
                              row_lower_bounds=None, row_upper_bounds=None):
        """Collect a KnapsackCutGroup from each verified row.

        histories:        list of per-row dict[layer_name -> (loc, sign, beta)]
        output_lbs:       post-CROWN final-layer lb, shape (batch,) or (batch, n_specs)
        rhs:              (batch, n_specs) tensor. Row j verified iff
                          any(lb[j] > rhs[j]) i.e. ~all(<=).
        row_lower_bounds: optional dict[layer_name -> tensor[batch, ...]]
                          per-row preact lower bounds. If omitted, falls back
                          to live self._net[name].lower (must match batch).
        row_upper_bounds: optional dict[layer_name -> tensor[batch, ...]]
        """
        if not self._initialized:
            return

        if isinstance(output_lbs, torch.Tensor):
            if output_lbs.ndim == 1:
                verified_mask = output_lbs > rhs
            else:
                verified_mask = ~torch.all(output_lbs <= rhs, dim=1)
            v_idx = torch.where(verified_mask)[0].tolist()
        else:
            v_idx = []

        if self._debug:
            try:
                mx = output_lbs.max().item()
            except Exception:
                mx = None
            print(
                f"[knapsack-dbg] collect: lb_final shape={tuple(getattr(output_lbs, 'shape', ()))} "
                f"max={mx} verified_rows={len(v_idx)}"
            )

        if not v_idx or histories is None:
            return

        bm = self._net

        for j in v_idx:
            if j >= len(histories):
                continue
            hist = histories[j]

            fixed = []
            for key in list(hist.keys()):
                if key not in self._key_mapping:
                    continue
                if key not in self._preact_name_to_relu:
                    continue
                loc_t, sign_t = self._hist_to_tensors(hist[key][0], hist[key][1])
                if loc_t is None:
                    continue
                layer_idx = self._key_mapping[key]
                n_fixed = loc_t.numel()
                for i in range(n_fixed):
                    phase = int(sign_t[i].item())
                    if phase not in (-1, 1):
                        continue
                    fixed.append((layer_idx, int(loc_t[i].item()), phase == 1))

            if not fixed:
                continue

            seen_layers = {L for (L, _, _) in fixed}

            group = KnapsackCutGroup()
            group.depth = len(fixed)
            for L, n, active in fixed:
                ds = self._downstream_relus.get(L, set())
                if any(L2 in ds for L2 in seen_layers if L2 != L):
                    continue  # not furthest
                relu_op = self._layer_idx_to_relu.get(L)
                if relu_op is None:
                    continue
                preact_name = relu_op.inputs[0].name

                # Threshold = leaf's per-row bound on pre_b.
                bounds_tensor = None
                if active and row_lower_bounds is not None and preact_name in row_lower_bounds:
                    bounds_tensor = row_lower_bounds[preact_name]
                elif (not active) and row_upper_bounds is not None and preact_name in row_upper_bounds:
                    bounds_tensor = row_upper_bounds[preact_name]
                else:
                    preact_node = bm[preact_name]
                    bounds_tensor = preact_node.lower if active else preact_node.upper

                if not isinstance(bounds_tensor, torch.Tensor):
                    if self._debug:
                        print(f"[knapsack-dbg] collect: preact {preact_name} non-tensor bounds")
                    continue
                if j >= bounds_tensor.shape[0]:
                    continue
                threshold = float(bounds_tensor[j].flatten()[n].item())
                cut = self.build_cut(relu_op, n, active, threshold)
                if cut is None:
                    continue
                group.cuts.append(cut)
                self._num_cuts_built += 1
                if self._debug:
                    print(
                        f"[knapsack-dbg] BUILD cut: target_pre={cut.target_b_var} "
                        f"(L={cut.relu_layer_idx}, j={cut.neuron_idx}) "
                        f"phase={'ACTIVE' if cut.is_active else 'INACTIVE'} "
                        f"constant={cut.constant:.6g} threshold={cut.threshold:.6g} "
                        f"weights={len(cut.coefficients)}"
                    )
                    self._self_check(cut, j, row_lower_bounds, row_upper_bounds)

            if group.cuts:
                self._cut_groups.append(group)

    def _self_check(self, cut, j, row_lower_bounds, row_upper_bounds):
        """Evaluate envelope using THIS leaf row's bounds. By construction
        envelope should match or be tighter than threshold."""
        bm = self._net
        env = cut.constant
        for var, w in cut.coefficients.items():
            if var[0] == "input":
                ptb = getattr(bm[bm.input_name[0]], "perturbation", None)
                if ptb is None:
                    continue
                lo = float(ptb.x_L.reshape(-1)[var[1]].item())
                hi = float(ptb.x_U.reshape(-1)[var[1]].item())
            else:
                L_var, n_var = var
                preact = self._layer_idx_to_relu[L_var].inputs[0]
                if row_lower_bounds is not None and preact.name in row_lower_bounds:
                    pre_lb_row = row_lower_bounds[preact.name][j].reshape(-1)[n_var]
                    pre_ub_row = row_upper_bounds[preact.name][j].reshape(-1)[n_var]
                else:
                    pre_lb_row = bm[preact.name].lower[j].reshape(-1)[n_var]
                    pre_ub_row = bm[preact.name].upper[j].reshape(-1)[n_var]
                lo = max(0.0, float(pre_lb_row.item()))
                hi = max(0.0, float(pre_ub_row.item()))
            if cut.is_active:
                env += w * lo if w > 0 else w * hi
            else:
                env += w * hi if w > 0 else w * lo
        if cut.is_active:
            ok = env >= cut.threshold - 1e-4
            kind = "ACTIVE env>=thr"
        else:
            ok = env <= cut.threshold + 1e-4
            kind = "INACTIVE env<=thr"
        print(
            f"[knapsack-dbg]   self-check: env={env:.6g} thr={cut.threshold:.6g} "
            f"({kind}) {'OK' if ok else 'FAIL'}"
        )

    # ---------- pruning check ----------

    def check_pruning(self, batch_size, interm_bounds=None, input_ptb=None):
        """Return bool[batch_size] mask: True = row implied by some cut group.

        interm_bounds: dict[layer_name, [lb_tensor, ub_tensor]] OR
                       dict[layer_name, (lb, ub)] OR
                       two separate dicts via row_lower_bounds/row_upper_bounds
                       — caller passes None to fall back to net state.
        """
        if not self._initialized or not self._cut_groups:
            return None

        bm = self._net

        layer_lb = {}
        layer_ub = {}
        if interm_bounds is not None:
            for name, pair in interm_bounds.items():
                if name not in self._key_mapping:
                    continue
                lb, ub = pair[0], pair[1]
                if not isinstance(lb, torch.Tensor) or not isinstance(ub, torch.Tensor):
                    continue
                if lb.shape[0] != batch_size or ub.shape[0] != batch_size:
                    continue
                layer_idx = self._key_mapping[name]
                layer_lb[layer_idx] = lb.reshape(batch_size, -1)
                layer_ub[layer_idx] = ub.reshape(batch_size, -1)
        else:
            for r in self._relus:
                preact_name = r.inputs[0].name
                node = bm[preact_name]
                if not isinstance(node.lower, torch.Tensor):
                    continue
                if node.lower.shape[0] != batch_size:
                    continue
                layer_idx = self._key_mapping[preact_name]
                layer_lb[layer_idx] = node.lower.reshape(batch_size, -1)
                layer_ub[layer_idx] = node.upper.reshape(batch_size, -1)

        input_lb_flat = None
        input_ub_flat = None
        ptb = input_ptb
        if ptb is None:
            try:
                input_node = bm[bm.input_name[0]]
                ptb = getattr(input_node, "perturbation", None)
            except Exception:
                ptb = None
        if ptb is not None:
            if hasattr(ptb, "x_L") and ptb.x_L is not None:
                input_lb_flat = ptb.x_L.reshape(-1)
            if hasattr(ptb, "x_U") and ptb.x_U is not None:
                input_ub_flat = ptb.x_U.reshape(-1)

        device = "cpu"
        for t in layer_lb.values():
            device = t.device
            break

        implied = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._num_prune_checks_rows += batch_size

        for group in self._cut_groups:
            group_implied = torch.ones(batch_size, dtype=torch.bool, device=device)
            for cut in group.cuts:
                envelope = torch.full(
                    (batch_size,), cut.constant, dtype=torch.float32, device=device
                )
                cut_ok = True
                for var, w in cut.coefficients.items():
                    if var[0] == "input":
                        if input_lb_flat is None or input_ub_flat is None:
                            cut_ok = False
                            break
                        lb = float(input_lb_flat[var[1]].item())
                        ub = float(input_ub_flat[var[1]].item())
                        if cut.is_active:
                            contrib = w * lb if w > 0 else w * ub
                        else:
                            contrib = w * ub if w > 0 else w * lb
                        envelope = envelope + contrib
                    else:
                        L, n = var
                        if L not in layer_lb:
                            cut_ok = False
                            break
                        pre_lb = layer_lb[L][:, n]
                        pre_ub = layer_ub[L][:, n]
                        lb_fvar = torch.clamp(pre_lb, min=0.0)
                        ub_fvar = torch.clamp(pre_ub, min=0.0)
                        if cut.is_active:
                            contrib = w * lb_fvar if w > 0 else w * ub_fvar
                        else:
                            contrib = w * ub_fvar if w > 0 else w * lb_fvar
                        envelope = envelope + contrib
                if not cut_ok:
                    group_implied = torch.zeros_like(group_implied)
                    break
                if cut.is_active:
                    cut_implied_row = envelope >= cut.threshold
                else:
                    cut_implied_row = envelope <= cut.threshold
                group_implied = group_implied & cut_implied_row
            implied = implied | group_implied

        n_pruned = int(implied.sum().item())
        self._num_prunes_rows += n_pruned
        if self._debug:
            print(
                f"[knapsack-dbg] check_pruning: {n_pruned}/{batch_size} rows "
                f"implied across {len(self._cut_groups)} groups "
                f"({self._num_prunes_rows} cumulative prunes)"
            )
        return implied

    def print_summary(self):
        if not self._initialized:
            return
        print(
            f"[knapsack] summary: groups={len(self._cut_groups)} "
            f"cuts_built={self._num_cuts_built} "
            f"prune_checks_rows={self._num_prune_checks_rows} "
            f"prunes_rows={self._num_prunes_rows}"
        )

    @property
    def num_cut_groups(self):
        return len(self._cut_groups)
