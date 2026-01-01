import argparse
import torch
import time
import os
import random

import gc
from helper.network.read_onnx import parse_onnx
from helper.network.read_pth import parse_pth

from helper.spec.objective import parse_vnnlib

from helper.misc.logger import logger, LOGGER_LEVEL
from helper.misc.export import get_adv_string

from verifier.verifier import Verifier 

from setting import Settings

from heuristic.sat_solver import SATSolver

def _resolve_with_bab(objectives, preconditions, time_limit):
    """ Reverify using BaB with the shortened time limit, used to find UNSAT cores even after SAT is found """
    logger.info(f'[!] Re-solving with BaB for {time_limit} seconds to find UNSAT cores.')
    logger.debug(f'[DEBUG] time_limit parameter value: {time_limit}, type: {type(time_limit)}')

    # Create temp verifier instance
    temp_verifier = Verifier(
        net=model,
        input_shape=input_shape,
        batch=args.batch,
        device=args.device,
    )

    # Re-solve forcing BaB with time limit, resolve_for_cores=True skips preprocessing
    status = temp_verifier.verify(
        objectives,
        preconditions=incremental_preconditions,
        timeout=time_limit,
        force_split='hidden',
        disable_attack=True,
        resolve_for_cores=True
    )

    # Collect new cores
    new_cores = []

    for v in temp_verifier.all_conflict_clauses.values():
        new_cores.extend(v)

    logger.info(f'[!] Re-solving found {len(new_cores)} new UNSAT cores.')
    
    # clean up verifier
    del temp_verifier
    if 'cuda' in args.device:
        gc.collect()
        torch.cuda.empty_cache()

    return new_cores

def _minimize_core(objectives, condition, time_limit=100, split_impact_stats=None, removal_threshold=0.2, random_drop_percentage=0.5):
    """ Minimize the UNSAT core by removing literals and checking if still UNSAT """  
    if split_impact_stats is None or len(split_impact_stats) == 0:
        logger.info(f'[!] No split_impact_stats available, skipping minimization')
        return None
    
    # Create temp verifier instance
    temp_verifier = Verifier(
        net=model,
        input_shape=input_shape,
        batch=args.batch,
        device=args.device,
    )
    
    # Initialize domains_list by doing a minimal verify call to set up var_mapping
    try:
        # Initialize by doing a minimal verify - this will set up domains_list
        _ = temp_verifier.verify(
            objectives,
            preconditions=[],
            timeout=0.1,
            disable_attack=True
        )
    except:
        pass
    
    # Get reversed_var_mapping to map literals to (layer_name, neuron_id)
    if not hasattr(temp_verifier, 'domains_list') or temp_verifier.domains_list is None:
        logger.warning(f'[!] Could not initialize domains_list for var_mapping, skipping minimization')
        del temp_verifier
        if 'cuda' in args.device:
            gc.collect()
            torch.cuda.empty_cache()
        return None
    
    # Convert condition to conflict clause if it's a history dict
    if isinstance(condition, dict):
        from heuristic.util import _history_to_conflict_clause
        conflict_clause = _history_to_conflict_clause(condition, temp_verifier.domains_list.var_mapping)
        if len(conflict_clause) == 0:
            logger.warning(f'[!] Could not convert history to conflict clause, skipping minimization')
            del temp_verifier
            if 'cuda' in args.device:
                gc.collect()
                torch.cuda.empty_cache()
            return None
    else:
        conflict_clause = condition
    
    reversed_var_mapping = temp_verifier.domains_list.reversed_var_mapping
    
    minimized_core = []
    dropped_literals = []
    
    # determine literals to drop based on impact statistics
    for literal in conflict_clause:
        # Map literal to (layer_name, neuron_id)
        abs_literal = abs(literal)
        if abs_literal not in reversed_var_mapping:
            # Literal not in mapping, keep it to be safe
            logger.debug(f'[DEBUG] Literal {literal} not found in var_mapping, keeping it')
            minimized_core.append(literal)
            continue
        
        layer_name, neuron_id = reversed_var_mapping[abs_literal]
        split_key = (layer_name, neuron_id)
        
        # Look up impact stats
        if split_key not in split_impact_stats:
            # No stats for this split, keep the literal
            logger.debug(f'[DEBUG] No impact stats for ({layer_name}, {neuron_id}), keeping literal {literal}')
            minimized_core.append(literal)
            continue
        
        stats = split_impact_stats[split_key]
        fixes = stats.get('fixes', 0)
        potential_fixes = stats.get('potential_fixes', 1)
        
        # Calculate fixes over potential fixes ratio
        if potential_fixes > 0:
            ratio = fixes / potential_fixes
        else:
            ratio = 0.0
        
        # Log the fixes over potential fixes
        logger.info(f'[MINIMIZE] Literal {literal} -> ({layer_name}, {neuron_id}): fixes={fixes}, potential_fixes={potential_fixes}, ratio={ratio:.4f}')
        
        # if the fixes over potential fixes is under removal_threshold
        if ratio < removal_threshold:
            # don't add the literal
            dropped_literals.append((literal, layer_name, neuron_id, ratio))
            logger.debug(f'[MINIMIZE] Dropping literal {literal} (ratio {ratio:.4f} < threshold {removal_threshold})')
        else:
            # otherwise add the literal
            minimized_core.append(literal)
            logger.debug(f'[MINIMIZE] Keeping literal {literal} (ratio {ratio:.4f} >= threshold {removal_threshold})')
    
    # print the minimized core vs original core
    logger.info(f'[MINIMIZE] Original core size: {len(conflict_clause)}, Minimized core size: {len(minimized_core)}, Dropped: {len(dropped_literals)}')
    
    if len(minimized_core) == 0:
        logger.warning(f'[MINIMIZE] All literals dropped, choosing random literals to keep')
        # randomly drop random_drop_percentage of literals from original cores
        num_to_drop = int(len(conflict_clause) * random_drop_percentage)
        drop_indices = set(random.sample(range(len(conflict_clause)), num_to_drop))
        minimized_core = [lit for i, lit in enumerate(conflict_clause) if i not in drop_indices]
    
    if len(minimized_core) >= len(conflict_clause):
        # No improvement, return None
        logger.info(f'[MINIMIZE] No improvement (minimized >= original), returning None')
        del temp_verifier
        if 'cuda' in args.device:
            gc.collect()
            torch.cuda.empty_cache()
        return None
    
    # fix cores in condition minus dropped literals
    status = temp_verifier.verify(
        objectives,
        preconditions=[minimized_core],
        timeout=time_limit,
        disable_attack=True
    )
    
    # if still UNSAT, return minimized core, else return None
    if status == 'unsat':
        logger.info(f'[MINIMIZE] Minimized core is still UNSAT, returning minimized core')

        # print minimized core
        logger.debug(f'[MINIMIZE] Minimized core literals: {minimized_core}')
        del temp_verifier
        if 'cuda' in args.device:
            gc.collect()
            torch.cuda.empty_cache()
        return minimized_core
    else:
        logger.info(f'[MINIMIZE] Minimized core is {status}, returning None (minimization failed)')
        del temp_verifier
        if 'cuda' in args.device:
            gc.collect()
            torch.cuda.empty_cache()
        return None

def _unit_propagate_cores(cores):
    """
    Apply unit propagation across all cores using the built-in SAT solver.
    
    Returns:
        tuple: (is_unsat, simplified_cores)
        - is_unsat: True if UNSAT is detected (conflicting unit cores or empty clause)
        - simplified_cores: List of simplified cores after unit propagation
    """
    if len(cores) == 0:
        return False, cores
    
    logger.info(f'[UNIT-PROP] Starting unit propagation on {len(cores)} cores')
    
    # Create SAT solver with all cores
    sat_solver = SATSolver(cores)
    
    # Perform BCP (Boolean Constraint Propagation / unit propagation)
    success, inferred_literals = sat_solver.bcp()
    
    if not success:
        # Conflict detected during BCP - this means UNSAT
        logger.info(f'[UNIT-PROP] CONFLICT detected during unit propagation! Problem is UNSAT.')
        return True, []
    
    if len(inferred_literals) > 0:
        logger.info(f'[UNIT-PROP] Inferred {len(inferred_literals)} unit literals: {inferred_literals[:10]}{"..." if len(inferred_literals) > 10 else ""}')
    
    # Check if we have an empty clause (all remaining clauses are satisfied)
    if len(sat_solver.clauses) == 0:
        logger.info(f'[UNIT-PROP] All clauses satisfied - this is actually SAT, not UNSAT')
        return False, []
    
    # Check for empty clauses (clauses where all literals were removed)
    empty_clause_mask = sat_solver.clauses.count_nonzero(dim=1) == 0
    if empty_clause_mask.any():
        logger.info(f'[UNIT-PROP] EMPTY CLAUSE detected! Problem is UNSAT.')
        return True, []
    
    # Convert remaining clauses back to list format
    simplified_cores = []
    for clause_tensor in sat_solver.clauses:
        # Remove zeros (removed literals) from the clause
        clause = [int(lit) for lit in clause_tensor if lit != 0]
        if len(clause) > 0:
            simplified_cores.append(clause)
    
    if len(inferred_literals) > 0:
        logger.info(f'[UNIT-PROP] Simplified cores: {len(cores)} -> {len(simplified_cores)}')
        total_literals_before = sum(len(c) for c in cores)
        total_literals_after = sum(len(c) for c in simplified_cores)
        logger.info(f'[UNIT-PROP] Total literals: {total_literals_before} -> {total_literals_after}')
    
    return False, simplified_cores

if __name__ == '__main__':
    START_TIME = time.time()

    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True,
                        help="load pretrained ONNX model from this specified path.")
    parser.add_argument('--spec', type=str, required=False,
                        help="path to VNNLIB specification file.")
    parser.add_argument('--incremental-specs', type=str, nargs='+', required=False,
                        help="path to VNNLIB specification file.")
    parser.add_argument('--input_shape', type=int, nargs='+', default=None,
                        help="Input shape of network, e.g., --input_shape 1 3 32 32")
    parser.add_argument('--output_shape', type=int, nargs='+', default=None,
                        help="Output shape of network, e.g., --output_shape 1 10")
    parser.add_argument('--batch', type=int, default=1000,
                        help="maximum number of branches to verify in each iteration")
    parser.add_argument('--timeout', type=float, default=3600,
                        help="timeout in seconds")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help="choose device to use for verifying.")
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=2, 
                        help='the logger level (0: NOTSET, 1: INFO, 2: DEBUG).')
    parser.add_argument('--result_file', type=str, required=False,
                        help="file to save execution results.")
    parser.add_argument('--export_cex', action='store_true',
                        help="enable exporting counter-example to result file.")
    parser.add_argument('--disable_attack', action='store_false',
                        help="disable attack.")
    parser.add_argument('--disable_restart', action='store_false',
                        help="disable RESTART heuristic.")
    parser.add_argument('--disable_stabilize', action='store_false',
                        help="disable STABILIZE heuristic.")
    parser.add_argument('--force_split', type=str, choices=['input', 'hidden'],
                        help="select SPLITTING strategy.")
    parser.add_argument('--reasoning_file', type=str, required=False,
                        help="file to save reasoning steps.")
    parser.add_argument('--setting_file', type=str, required=False,
                        help="file to load specific settings.")
    parser.add_argument('--test', action='store_true',
                        help="test on small example with special settings.")
    parser.add_argument('--export_runtime', action='store_true', required=False,
                        help="output runtime.")
    parser.add_argument('--resolve-time-limit', type=float, default=10, required=False,
                        help="time limit in seconds for re-solving with BaB to find UNSAT cores after SAT is found (default: 10)")
    
    args = parser.parse_args()

    if args.spec is None and args.incremental_specs is None:
        parser.error("one of the arguments --spec or --incremental-specs is required")
    
    specs = args.incremental_specs if args.incremental_specs else [args.spec]

    Settings.setup(args)
    print(Settings)
        
    # set device
    if not torch.cuda.is_available():
        args.device = 'cpu'
        
    # set logger level
    logger.setLevel(LOGGER_LEVEL[args.verbosity])
    
    # network
    if args.net.endswith('.onnx'):
        model, input_shape, output_shape = parse_onnx(args.net, args.input_shape, args.output_shape)
    elif args.net.endswith('.pth'):
        model, input_shape, output_shape = parse_pth(args.net, args.input_shape, args.output_shape)
    else:
        raise NotImplementedError('Unsupported network type')
    
    model.to(args.device)

    if args.verbosity:
        print(model)
    logger.info(f'[!] Input shape: {input_shape}')
    logger.info(f'[!] Output shape: {output_shape}')

    # remove result file if exists
    if args.result_file and os.path.exists(args.result_file):
        os.remove(args.result_file)

    # define the incremental preconditions list
    incremental_preconditions = []

    for i, spec in enumerate(specs):
        logger.info(f'[!] Verifying spec: {spec}')
        
        # specification
        objectives = parse_vnnlib(spec, input_shape)

        # verifier
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=args.batch,
            device=args.device,
        )
        
        # verify
        START_TIME = time.time()
        status = verifier.verify(objectives, preconditions=incremental_preconditions, timeout=args.timeout, force_split=args.force_split)
        runtime = time.time() - START_TIME

        # collect new preconditions
        new_preconditions = []
        for v in verifier.all_conflict_clauses.values():
            # Convert histories to conflict clauses if needed
            from heuristic.util import _history_to_conflict_clause
            for history in v:
                if isinstance(history, dict):
                    # History dict needs to be converted to conflict clause
                    if hasattr(verifier, 'domains_list') and hasattr(verifier.domains_list, 'var_mapping'):
                        clause = _history_to_conflict_clause(history, verifier.domains_list.var_mapping)
                        if len(clause) > 0:
                            new_preconditions.append(clause)
                else:
                    # Already a conflict clause
                    new_preconditions.append(history)
        incremental_preconditions.extend(new_preconditions)
        verifier.all_conflict_clauses = {} # clear for next run
        logger.info(f'[!] Transferred {len(new_preconditions)} UNSAT cores')

        # minimize preconditions
        if len(incremental_preconditions) > 0:
            # get shortest conflict clauses 
            incremental_preconditions.sort(key=lambda x: len(x), reverse=False)

            # minimize three shortest conflict clauses 
            for condition in incremental_preconditions[:3]:
                minimized_core = _minimize_core(objectives, condition, split_impact_stats=verifier.split_impact_stats)

                if minimized_core is not None:
                    incremental_preconditions.remove(condition)
                    incremental_preconditions.append(minimized_core)
                    logger.info(f'[!] Minimized a core from size {len(condition)} to size {len(minimized_core)}')
            
            # Apply unit propagation across all cores
            is_unsat, simplified_cores = _unit_propagate_cores(incremental_preconditions)
            
            if is_unsat:
                # UNSAT detected through unit propagation!
                logger.info(f'[!] UNSAT detected through unit propagation of cores!')
                logger.info(f'[!] Property {i+1}/{len(specs)} is UNSAT. All remaining properties are also UNSAT.')
                
                # Mark current property as UNSAT
                status = 'unsat'
                runtime = time.time() - START_TIME
                
                if args.result_file:
                    with open(args.result_file, 'a') as fp:
                        if args.export_runtime:
                            print(f'unsat,{runtime:.04f}', file=fp)
                        else:
                            print('unsat', file=fp)
                
                logger.info(f'[!] Result: unsat')
                logger.info(f'[!] Runtime: {runtime:.04f}')
                print(f'unsat,{runtime:.04f}')
                
                # Mark all remaining properties as UNSAT
                for j in range(i + 1, len(specs)):
                    logger.info(f'[!] Verifying spec: {specs[j]}')
                    logger.info(f'[!] Property is UNSAT due to unit propagation conflict in property {i+1} (incremental verification).')
                    logger.info(f'[!] Result: unsat')
                    logger.info(f'[!] Runtime: 0.0000')
                    
                    if args.result_file:
                        with open(args.result_file, 'a') as fp:
                            if args.export_runtime:
                                print(f'unsat,0.0000', file=fp)
                            else:
                                print('unsat', file=fp)
                    
                    print(f'unsat,0.0000')
                
                # Exit the loop since all remaining properties are UNSAT
                break
            else:
                # Update cores with simplified versions
                incremental_preconditions = simplified_cores
        
        # output
        logger.info(f'[!] Iterations: {verifier.iteration}')
        if verifier.adv is not None:
            logger.info(f'adv (first 5): {verifier.adv.flatten()[:5].detach().cpu()}')
            logger.debug(f'output: {verifier.net(verifier.adv).flatten().detach().cpu()}')
            
        # export
        if args.result_file:
            with open(args.result_file, 'a') as fp:
                if args.export_runtime:
                    print(f'{status},{runtime:.04f}', file=fp)
                else:
                    print(status, file=fp)
                if (verifier.adv is not None) and args.export_cex:
                    print(get_adv_string(inputs=verifier.adv, net_path=args.net), file=fp)

        if args.reasoning_file and Settings.use_save_reasoning_step:
            if hasattr(verifier, 'domains_list') and not isinstance(verifier.domains_list, list):
                verifier.domains_list.reasoning_domains.export(args.reasoning_file)
            else:
                print(f'[!] Does not have any reasoning step')

        logger.info(f'[!] Result: {status}')
        logger.info(f'[!] Runtime: {runtime:.04f}')


        # clean up verifier
        del verifier
        if 'cuda' in args.device:
            gc.collect()
            torch.cuda.empty_cache()

        # if condition is UNSAT, all subsequent properties are also UNSAT
        if status == 'unsat':
            logger.info(f'[!] Property {i+1}/{len(specs)} is UNSAT. All remaining properties are also UNSAT.')
            
            # Mark all remaining properties as UNSAT
            for j in range(i + 1, len(specs)):
                logger.info(f'[!] Verifying spec: {specs[j]}')
                logger.info(f'[!] Property is UNSAT due to previous property {i+1} being UNSAT (incremental verification).')
                logger.info(f'[!] Result: unsat')
                logger.info(f'[!] Runtime: 0.0000')
                
                if args.result_file:
                    with open(args.result_file, 'a') as fp:
                        if args.export_runtime:
                            print(f'unsat,0.0000', file=fp)
                        else:
                            print('unsat', file=fp)
                
                print(f'unsat,0.0000')
            
            # Exit the loop since all remaining properties are UNSAT
            break

        # if condition is SAT, reverify under time limit
        if status == 'sat' and args.resolve_time_limit > 0:
            # Reverify using BaB with time limit
            logger.debug(f'[DEBUG] Calling _resolve_with_bab with args.resolve_time_limit={args.resolve_time_limit}')
            new_cores = _resolve_with_bab(objectives=objectives, preconditions=incremental_preconditions, time_limit=args.resolve_time_limit)

            # Add new cores to preconditions
            incremental_preconditions.extend(new_cores)
            
            # Apply unit propagation after adding new cores
            if len(incremental_preconditions) > 0:
                is_unsat, simplified_cores = _unit_propagate_cores(incremental_preconditions)
                
                if is_unsat:
                    # UNSAT detected through unit propagation after resolve!
                    logger.info(f'[!] UNSAT detected through unit propagation after resolve!')
                    logger.info(f'[!] Property {i+1}/{len(specs)} is actually UNSAT. All remaining properties are also UNSAT.')
                    
                    # Update status to unsat
                    status = 'unsat'
                    
                    # Update result file
                    if args.result_file:
                        with open(args.result_file, 'a') as fp:
                            if args.export_runtime:
                                print(f'unsat,{runtime:.04f}', file=fp)
                            else:
                                print('unsat', file=fp)
                    
                    logger.info(f'[!] Result: unsat (corrected from sat via unit propagation)')
                    print(f'unsat,{runtime:.04f}')
                    
                    # Mark all remaining properties as UNSAT
                    for j in range(i + 1, len(specs)):
                        logger.info(f'[!] Verifying spec: {specs[j]}')
                        logger.info(f'[!] Property is UNSAT due to unit propagation conflict in property {i+1} (incremental verification).')
                        logger.info(f'[!] Result: unsat')
                        logger.info(f'[!] Runtime: 0.0000')
                        
                        if args.result_file:
                            with open(args.result_file, 'a') as fp:
                                if args.export_runtime:
                                    print(f'unsat,0.0000', file=fp)
                                else:
                                    print('unsat', file=fp)
                        
                        print(f'unsat,0.0000')
                    
                    # Exit the loop since all remaining properties are UNSAT
                    break
                else:
                    # Update cores with simplified versions
                    incremental_preconditions = simplified_cores

        print(f'{status},{runtime:.04f}')


