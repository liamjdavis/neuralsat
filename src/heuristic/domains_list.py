from __future__ import annotations
from collections import defaultdict
from beartype import beartype
import typing
import torch
import time
import os

if typing.TYPE_CHECKING:
    import abstractor

from helper.proof.reasoning_domains import ReasoningDomains
from helper.misc.tensor_storage import TensorStorage
from helper.misc.result import AbstractResults
from helper.misc.logger import logger

from abstractor.utils import _copy_history
from heuristic.util import compute_masks
from setting import Settings


class DomainsList:
    
    "List of unverified branches"

    @beartype
    def __init__(self: 'DomainsList', 
                 net: 'abstractor.auto_LiRPA.BoundedModule',
                 objective_ids: torch.Tensor,
                 output_lbs: torch.Tensor,
                 input_lowers: torch.Tensor, 
                 input_uppers: torch.Tensor, 
                 lower_bounds: dict | None, 
                 upper_bounds: dict | None, 
                 lAs: dict, 
                 histories: dict | None, 
                 slopes: dict, 
                 cs: torch.Tensor, 
                 rhs: torch.Tensor, 
                 input_split: bool = False, 
                 preconditions: dict = {},
                 resolve_for_cores: bool = False) -> None:

        self.net = net
        self.final_name = self.net.final_node_name

        self.input_split = input_split
        self.visited = len(input_lowers)
        self.all_conflict_clauses = {int(_): [] for _ in objective_ids}
        self.use_restart = Settings.use_restart and (lower_bounds is not None) and (not input_split)
        
        # Global UNSAT core cache for cross-property incremental verification
        # Initialized empty; populated by save_conflict_clauses when domains prove UNSAT
        self.global_unsat_cores = []  # List[List[int]] - SAT clauses learned from all properties
        
        # Populate global_unsat_cores with initial preconditions
        if len(preconditions) > 0:
            # preconditions is {obj_id: [clauses/histories]}
            # We iterate over the first objective's preconditions since they are duplicated across all objectives
            first_key = next(iter(preconditions))
            
            # Helper to convert histories to clauses if needed
            from heuristic.util import _history_to_conflict_clause
            
            for cond in preconditions[first_key]:
                if isinstance(cond, list):
                    # Already a clause
                    self.global_unsat_cores.append(cond)
                elif isinstance(cond, dict):
                    # Convert history dict to clause
                    # Note: We need var_mapping, which is a property that computes on demand
                    clause = _history_to_conflict_clause(cond, self.var_mapping)
                    if len(clause) > 0:
                        self.global_unsat_cores.append(clause)
        
        # unverified indices 
        if resolve_for_cores:
            # When resolving for cores, keep all objectives even if already verified
            remain_idx = torch.arange(len(output_lbs))
            logger.info(f'[resolve_for_cores=True] Keeping all {len(remain_idx)} objectives for UNSAT core collection')
        else:
            remain_idx = torch.where((output_lbs.detach().cpu() <= rhs.detach().cpu()).all(1))[0]
        
        # decisions
        all_histories = [_copy_history(histories) for _ in range(len(cs))] if not input_split else None
        all_betas = [None for i in range(len(cs))] if not input_split else None
        
        # sat solvers
        self.all_sat_solvers = None
        if self.use_restart and any([len(_) > 0 for _ in preconditions.values()]):
            tic = time.time()
            remain_idx = self.init_sat_solver(
                objective_ids=objective_ids,
                lower_bounds=lower_bounds, 
                upper_bounds=upper_bounds, 
                histories=all_histories, 
                preconditions=preconditions,
                remain_idx=remain_idx,
            )
            logger.info(f'Initialized {len(self.all_sat_solvers)} solvers with {len(preconditions)} learned clauses in {time.time() - tic:.03f} seconds')
        
        # objective indices
        self.all_objective_ids = TensorStorage(objective_ids[remain_idx].cpu())
        
        # input bounds
        self.all_input_lowers = TensorStorage(input_lowers[remain_idx].cpu())
        self.all_input_uppers = TensorStorage(input_uppers[remain_idx].cpu())
        
        # output bounds
        self.all_output_lowers = TensorStorage(output_lbs[remain_idx].cpu())
        
        # properties
        self.all_cs = TensorStorage(cs[remain_idx].cpu())
        self.all_rhs = TensorStorage(rhs[remain_idx].cpu())
    
        # alpha
        self.all_slopes = defaultdict(dict)
        for k in slopes:
            self.all_slopes[k] = {}
            for kk, v in slopes[k].items():
                if kk not in self.all_slopes[k]:
                    self.all_slopes[k][kk] = TensorStorage(v[:, :, remain_idx].cpu(), concat_dim=2)
                else:
                    self.all_slopes[k][kk].append(v[:, :, remain_idx].cpu())

        # lAs
        self.all_lAs = {k: TensorStorage(v[remain_idx].cpu()) for k, v in lAs.items()} if len(lAs) else None
        
        if self.input_split:
            self.all_lower_bounds = self.all_upper_bounds = self.all_histories = self.all_betas = None
            if Settings.use_save_reasoning_step:
                # hidden bounds
                self.all_lower_bounds = {k: TensorStorage(v[remain_idx].cpu()) for k, v in lower_bounds.items() if k != self.final_name}
                self.all_upper_bounds = {k: TensorStorage(v[remain_idx].cpu()) for k, v in upper_bounds.items() if k != self.final_name}
        else: # hidden spliting 
            # hidden bounds
            self.all_lower_bounds = {k: TensorStorage(v[remain_idx].cpu()) for k, v in lower_bounds.items() if k != self.final_name}
            self.all_upper_bounds = {k: TensorStorage(v[remain_idx].cpu()) for k, v in upper_bounds.items() if k != self.final_name}
            # decisions
            self.all_histories = [all_histories[_] for _ in remain_idx]
            self.all_betas = [all_betas[_] for _ in remain_idx]
            
        # proof
        if Settings.use_save_reasoning_step:
            # net architecture
            input_name = [_.name for _ in self.net.roots() if _.perturbed]
            assert len(input_name) == 1
            net_info = {
                'input_name': input_name[0],
                'output_name': self.final_name,
                'graph': []
            }
            for n in self.net.nodes():
                if n.perturbed:
                    net_info['graph'].append({n.name: [_.name for _ in n.inputs if _.perturbed]})

            # reasoning domains
            self.reasoning_domains = ReasoningDomains(
                net_info=net_info,
                objective_ids=objective_ids,
                input_lowers=input_lowers,
                input_uppers=input_uppers,
                output_lbs=output_lbs,
                lower_bounds={k: v for k, v in lower_bounds.items() if k != self.final_name},
                upper_bounds={k: v for k, v in upper_bounds.items() if k != self.final_name},
                cs=cs,
                rhs=rhs,
                histories=all_histories,
                input_split=input_split,
                select_index=torch.tensor([i for i in range(len(input_lowers)) if i not in remain_idx]).int(),
            )
        
        self._check_consistent()
        
        
    @beartype
    @property
    def var_mapping(self: 'DomainsList') -> dict:
        if not hasattr(self, '_var_mapping'):
            self._var_mapping = {}
            count = 1
            for layer in self.net.split_nodes:
                for nid in range(layer.lower.flatten(start_dim=1).shape[-1]):
                    self._var_mapping[layer.name, nid] = count
                    count += 1
        return self._var_mapping
    
    
    @beartype
    @property
    def reversed_var_mapping(self: 'DomainsList') -> dict:
        if not hasattr(self, '_reversed_var_mapping'):
            self._reversed_var_mapping = {v: k for k, v in self.var_mapping.items()}
        return self._reversed_var_mapping
    
        
    @beartype
    def _check_consistent(self: 'DomainsList') -> None:
        # print('Checking domains:', len(self))
        assert len(self.all_input_lowers) == len(self.all_input_uppers) == len(self.all_output_lowers) == len(self), \
            print(len(self.all_input_lowers), len(self.all_input_uppers), len(self.all_output_lowers), len(self))
        assert len(self.all_cs) == len(self.all_rhs) == len(self.all_objective_ids) == len(self), \
            print(len(self.all_cs), len(self.all_rhs), len(self.all_objective_ids))
        assert all([vv.data.shape[2] == len(self) for v in self.all_slopes.values() for vv in v.values()]), \
            print([vv.data.shape[2] for v in self.all_slopes.values() for vv in v.values()], len(self))
        
        if self.all_lAs is not None:
            assert all([len(_) == len(self) for _ in self.all_lAs.values()])
        
        if self.all_lower_bounds is not None:
            assert len(self.all_lower_bounds) == len(self.all_upper_bounds) 
            assert all([len(_) == len(self) for _ in self.all_lower_bounds.values()])
            assert all([len(_) == len(self) for _ in self.all_upper_bounds.values()])
            
        if not self.input_split:
            assert len(self.all_betas) == len(self.all_histories) == len(self)
            if self.all_sat_solvers is not None:
                assert len(self.all_sat_solvers) == len(self), print(f'len(self.all_sat_solvers)={len(self.all_sat_solvers)}, len(self)={len(self)}')


    @beartype
    def pick_out(self: 'DomainsList', batch: int, device: str = 'cpu') -> AbstractResults:
        assert batch > 0
        batch = min(len(self), batch)
        self.visited += batch

        if torch.cuda.is_available(): 
            torch.cuda.synchronize()

        # input bounds
        new_input_lowers = self.all_input_lowers.pop(batch).to(device=device, non_blocking=True)
        new_input_uppers = self.all_input_uppers.pop(batch).to(device=device, non_blocking=True)
        
        # objective indices
        new_objective_ids = self.all_objective_ids.pop(batch).to(device='cpu')
        
        # output bounds
        new_output_lowers = self.all_output_lowers.pop(batch).to(device=device, non_blocking=True)
        
        # properties
        new_cs = self.all_cs.pop(batch).to(device=device, non_blocking=True)
        new_rhs = self.all_rhs.pop(batch).to(device=device, non_blocking=True)
        
        # alpha
        new_slopes = defaultdict(dict)
        for k, v in self.all_slopes.items():
            new_slopes[k] = {kk: vv.pop(batch).to(device=device, non_blocking=True) for (kk, vv) in v.items()}
            
        # lAs
        new_lAs = {k: lA.pop(batch).to(device=device, non_blocking=True) for (k, lA) in self.all_lAs.items()} if self.all_lAs is not None else None
    
        # hidden spliting 
        new_lower_bounds = {k: lb.pop(batch).to(device=device, non_blocking=True) for (k, lb) in self.all_lower_bounds.items()} if self.all_lower_bounds is not None else None
        new_upper_bounds = {k: ub.pop(batch).to(device=device, non_blocking=True) for (k, ub) in self.all_upper_bounds.items()} if self.all_upper_bounds is not None else None
        
        if self.input_split:
            # input splitting
            new_masks = new_betas = new_histories = None
            new_sat_solvers = None
        else: 
            # pop batch
            new_betas = self.all_betas[-batch:]
            new_histories = self.all_histories[-batch:]

            # remove batch
            self.all_betas = self.all_betas[:-batch]
            self.all_histories = self.all_histories[:-batch]
            
            if self.all_sat_solvers is not None:
                new_sat_solvers = self.all_sat_solvers[-batch:]
                # new_sat_solvers = copy.deepcopy(self.all_sat_solvers[-batch:])
                self.all_sat_solvers = self.all_sat_solvers[:-batch]
            else:
                new_sat_solvers = None
            
            new_masks = compute_masks(
                lower_bounds=new_lower_bounds, 
                upper_bounds=new_upper_bounds, 
                device=device,
            )
            
            assert len(new_betas) == len(new_histories) == batch
            assert len(new_input_lowers) == len(new_input_lowers) == batch
            assert len(new_lower_bounds[list(new_lower_bounds.keys())[0]]) == batch
            assert len(new_upper_bounds[list(new_upper_bounds.keys())[0]]) == batch
            assert len(new_lAs[list(new_lAs.keys())[0]]) == batch 
        
        self._check_consistent()
        
        return AbstractResults(**{
            'objective_ids': new_objective_ids,
            'output_lbs': new_output_lowers,
            'input_lowers': new_input_lowers, 
            'input_uppers': new_input_uppers, 
            'masks': new_masks, 
            'lAs': new_lAs, 
            'lower_bounds': new_lower_bounds, 
            'upper_bounds': new_upper_bounds, 
            'slopes': new_slopes, 
            'betas': new_betas,
            'histories': new_histories,
            'cs': new_cs,
            'rhs': new_rhs,
            'sat_solvers': new_sat_solvers,
        })


    @beartype
    def add(self: 'DomainsList', domain_params: AbstractResults, decisions: list | torch.Tensor) -> None:
        # assert decisions is not None
        batch = len(domain_params.input_lowers)
        assert batch > 0
        
        # unverified indices
        remaining_index = torch.where((domain_params.output_lbs.detach().cpu() <= domain_params.rhs.detach().cpu()).all(1))[0]
        
        # hidden splitting
        if not self.input_split:
            # Global UNSAT core pruning (runs on every iteration, not just restarts)
            if hasattr(self, 'global_unsat_cores') and len(self.global_unsat_cores) > 0:
                logger.debug(f"[Global core] Checking {len(remaining_index)} domains against {len(self.global_unsat_cores)} cached cores")
                
                global_conflict_index = []
                for idx_ in remaining_index:
                    # Convert tensor index to int
                    idx_int = int(idx_.item()) if isinstance(idx_, torch.Tensor) else int(idx_)
                    
                    # extract fixed literals from bound propagation
                    fixed_literals = self._extract_fixed_literals_from_bounds(
                        domain_params.lower_bounds,
                        domain_params.upper_bounds,
                        idx_int
                    )
                    
                    if len(fixed_literals) > 0:
                        # create temporary SAT solver with global cores
                        from heuristic.sat_solver import SATSolver
                        temp_solver = SATSolver(self.global_unsat_cores)
                        
                        # check if current fixed literals conflict with cores
                        if not temp_solver.multiple_assign(fixed_literals):
                            global_conflict_index.append(idx_int)
                            logger.debug(f'[Global core] Pruned domain {idx_int} via assignment conflict')
                            continue
                        
                        # run BCP to detect implied conflicts
                        bcp_stat, _ = temp_solver.bcp()
                        if not bcp_stat:
                            global_conflict_index.append(idx_int)
                            logger.debug(f'[Global core] Pruned domain {idx_int} via BCP conflict')
                            continue
                
                # Remove globally-conflicted domains from remaining_index
                if len(global_conflict_index):
                    logger.info(f'[Global core] Pruned {len(global_conflict_index)} domains')
                    for gci in global_conflict_index:
                        remaining_index = remaining_index[remaining_index != gci]
            
            # Local SAT checking (restart-only, uses per-domain SAT solvers)
            if self.all_sat_solvers is not None:
                assert len(domain_params.sat_solvers) == batch
                assert decisions is not None
                
                extra_conflict_index = []
                for idx_ in remaining_index:
                    # local SAT check via boolean propagation
                    new_sat_solver = self.boolean_propagation(
                        domain_params=domain_params, 
                        decisions=decisions, 
                        batch_idx=idx_,
                    )
                    if new_sat_solver is None:
                        extra_conflict_index.append(idx_)
                        continue
                            
                    self.all_sat_solvers.append(new_sat_solver)
                    
                if len(extra_conflict_index):
                    logger.debug(f'[Local SAT] BCP removes {len(extra_conflict_index)} domains')
                    assert len(extra_conflict_index) == len(list(set(extra_conflict_index)))
                    for eci in extra_conflict_index:
                        remaining_index = remaining_index[remaining_index != eci]
            
            # decision histories
            self.all_histories.extend([domain_params.histories[i] for i in remaining_index])
            self.all_betas.extend([domain_params.betas[i] for i in remaining_index])
            
            # conflict clauses
            unsat_indices = torch.tensor([i for i in range(len(domain_params.input_lowers)) if i not in remaining_index]).int()
            self.save_conflict_clauses(
                domain_params=domain_params, 
                select_index=unsat_indices,
            )
            
            # Populate global UNSAT core cache for incremental verification
            # Convert decision trails of UNSAT domains to clauses and cache globally
            if hasattr(domain_params, 'histories') and domain_params.histories is not None:
                from heuristic.util import _history_to_conflict_clause
                for idx in unsat_indices:
                    if idx < len(domain_params.histories):
                        # Convert history (decision trail) to SAT clause
                        clause = _history_to_conflict_clause(
                            domain_params.histories[idx], 
                            self.var_mapping
                        )
                        if len(clause) > 0:
                            self.global_unsat_cores.append(clause)
                            logger.debug(f'[Global core] Cached UNSAT core with {len(clause)} literals')

        # hidden bounds
        if self.all_lower_bounds is not None:
            [v.append(domain_params.lower_bounds[k][remaining_index]) for k, v in self.all_lower_bounds.items()]
            [v.append(domain_params.upper_bounds[k][remaining_index]) for k, v in self.all_upper_bounds.items()]
        
        # objective indices
        self.all_objective_ids.append(domain_params.objective_ids[remaining_index])
        
        # input bounds
        self.all_input_lowers.append(domain_params.input_lowers[remaining_index])
        self.all_input_uppers.append(domain_params.input_uppers[remaining_index])
        
        # output bounds
        self.all_output_lowers.append(domain_params.output_lbs[remaining_index])
        
        # properties
        self.all_cs.append(domain_params.cs[remaining_index])
        self.all_rhs.append(domain_params.rhs[remaining_index])
        
        # alpha
        [vv.append(domain_params.slopes[k][kk][:,:,remaining_index]) for (k, v) in self.all_slopes.items() for (kk, vv) in v.items()]
            
        # lAs
        [v.append(domain_params.lAs[k][remaining_index]) for k, v in self.all_lAs.items()] if self.all_lAs is not None else None

        # proof
        if Settings.use_save_reasoning_step:
            self.reasoning_domains.add(
                domain_params=domain_params, 
                select_index=torch.tensor([i for i in range(len(domain_params.input_lowers)) if i not in remaining_index]).int(),
            )
        
        # checking
        self._check_consistent()
        

    @beartype
    def __len__(self: 'DomainsList') -> int:
        return len(self.all_input_lowers)


    @beartype
    @property
    def minimum_lowers(self: 'DomainsList') -> float:
        indices = (self.all_output_lowers - self.all_rhs).max(dim=1)[0].argsort()
        if len(indices):
            return (self.all_output_lowers[indices[0]] - self.all_rhs[indices[0]]).max().detach().item()
        return 1e-6


    @beartype
    def pick_out_worst_domains(self: 'DomainsList', batch: int, device: str = 'cpu') -> AbstractResults:
        indices = (self.all_output_lowers - self.all_rhs).max(dim=1)[0].argsort()[:batch]

        # objective ids
        new_objective_ids = self.all_objective_ids[indices].to(device=device, non_blocking=True)

        # input bounds
        new_input_lowers = self.all_input_lowers[indices].to(device=device, non_blocking=True)
        new_input_uppers = self.all_input_uppers[indices].to(device=device, non_blocking=True)

        # hidden bounds
        new_lower_bounds = {k: v[indices].to(device=device, non_blocking=True) for k, v in self.all_lower_bounds.items()}
        new_upper_bounds = {k: v[indices].to(device=device, non_blocking=True) for k, v in self.all_upper_bounds.items()}

        # output
        new_output_lowers = self.all_output_lowers[indices].to(device=device, non_blocking=True)
        
        # properties
        # new_cs = self.all_cs[indices].to(device=device, non_blocking=True)
        new_rhs = self.all_rhs[indices].to(device=device, non_blocking=True)
        
        self._check_consistent()
        
        return AbstractResults(**{
            'objective_ids': new_objective_ids,
            'input_lowers': new_input_lowers, 
            'input_uppers': new_input_uppers, 
            'lower_bounds': new_lower_bounds, 
            'upper_bounds': new_upper_bounds, 
            'output_lbs': new_output_lowers,
            # 'cs': new_cs,
            'rhs': new_rhs,
        })
        
        
    @beartype
    @torch.no_grad()
    def update_refined_bounds(self: 'DomainsList', domain_params: typing.Any) -> None:
        # updating
        for key in domain_params.lower_bounds:
            orig_shape = self.all_lower_bounds[key].size()[1:] # skip batch dim

            new_lower = torch.where(
                domain_params.lower_bounds[key].view(orig_shape) > self.all_lower_bounds[key].detach(), 
                domain_params.lower_bounds[key].view(orig_shape), 
                self.all_lower_bounds[key].detach()
            )
            self.all_lower_bounds[key].copy_(new_lower.detach())
            
            new_upper = torch.where(
                domain_params.upper_bounds[key].view(orig_shape) < self.all_upper_bounds[key].detach(),
                domain_params.upper_bounds[key].view(orig_shape), 
                self.all_upper_bounds[key].detach()
            )
            self.all_upper_bounds[key].copy_(new_upper.detach())
            
            if os.environ.get('NEURALSAT_ASSERT'):
                assert torch.all(self.all_lower_bounds[key].detach() <= self.all_upper_bounds[key].detach())

        # checking
        self._check_consistent()
        
    
    @beartype
    @torch.no_grad()
    def count_unstable_neurons(self: 'DomainsList') -> torch.Tensor | None:
        if self.all_lower_bounds is None:
            return None
        
        if not len(self):
            return None
        
        if not len(self.net.relus):
            return None
        
        new_masks = compute_masks(
            lower_bounds={k: v.data for k, v in self.all_lower_bounds.items()}, 
            upper_bounds={k: v.data for k, v in self.all_upper_bounds.items()}, 
            device='cpu',
        )
        # print(len(self), [_.shape for _ in new_masks.values()])
        n_unstable = sum([_.sum() for _ in new_masks.values()]).int()
        return n_unstable // len(self)

    
    @beartype
    def _extract_fixed_literals_from_bounds(self: 'DomainsList', lower_bounds: dict, 
                                            upper_bounds: dict, batch_idx: int) -> list[int]:
        """Extract SAT literals for neurons fixed by bound propagation.
        
        Args:
            lower_bounds: Lower bounds per layer
            upper_bounds: Upper bounds per layer
            batch_idx: Index of the domain to extract from
            
        Returns:
            List of SAT literals (positive = active, negative = inactive)
            
        Justification:
            Bound propagation may have fixed neurons beyond explicit branching decisions.
            These fixes are valid assumptions that must be consistent with UNSAT cores.
            By converting them to SAT literals, we can check if they conflict with
            previously learned cores before running expensive abstraction.
        """
        literals = []
        
        for layer_name in lower_bounds.keys():
            # Get bounds for this specific domain
            layer_lower = lower_bounds[layer_name][batch_idx].flatten().detach().cpu()
            layer_upper = upper_bounds[layer_name][batch_idx].flatten().detach().cpu()
            
            # Identify fixed neurons (ReLU assumption: fixed if bounds cross zero)
            for neuron_id in range(len(layer_lower)):
                if layer_lower[neuron_id] >= 0:
                    # Neuron provably active (lower bound >= 0)
                    var = self.var_mapping.get((layer_name, neuron_id))
                    if var is not None:
                        literals.append(var)
                        
                elif layer_upper[neuron_id] <= 0:
                    # Neuron provably inactive (upper bound <= 0)
                    var = self.var_mapping.get((layer_name, neuron_id))
                    if var is not None:
                        literals.append(-var)
        
        return literals
        
    from .util import init_sat_solver, update_hidden_bounds_histories, boolean_propagation, save_conflict_clauses
