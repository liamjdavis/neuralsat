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

    if hasattr(temp_verifier, 'domains_list') and hasattr(temp_verifier.domains_list, 'var_mapping'):
        from heuristic.util import _history_to_conflict_clause
        for v in temp_verifier.all_conflict_clauses.values():
            for history in v:
                if isinstance(history, dict):
                    clause = _history_to_conflict_clause(history, temp_verifier.domains_list.var_mapping)
                    if len(clause) > 0:
                        new_cores.append(clause)
                else:
                    new_cores.append(history)
    
    logger.info(f'[!] Re-solving found {len(new_cores)} new UNSAT cores.')
    
    # clean up verifier
    del temp_verifier
    if 'cuda' in args.device:
        gc.collect()
        torch.cuda.empty_cache()

    return new_cores

def _minimize_core(objectives, condition, time_limit=100, split_impact_stats=None, drop_threshold=0.25, random_drop_percentage=0.5):
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
    
    logger.info(f"[MINIMIZE] Core BEFORE minimization (Size {len(conflict_clause)}): {conflict_clause}")
    
    reversed_var_mapping = temp_verifier.domains_list.reversed_var_mapping
    
    minimized_core = []
    dropped_literals = []
    

    # Sort literals by impact (descending order, higher impact is better)
    # We want to drop the bottom `drop_threshold` percent.
    
    # First, collect all impacts
    literal_impacts = [] # (literal, impact, layer_name, neuron_id)
    
    for literal in conflict_clause:
        abs_literal = abs(literal)
        if abs_literal not in reversed_var_mapping:
            # Keep unmapped literals
            minimized_core.append(literal)
            continue
            
        layer_name, neuron_id = reversed_var_mapping[abs_literal]
        split_key = (layer_name, neuron_id)
        
        if split_key not in split_impact_stats:
            minimized_core.append(literal)
            continue
            
        stats = split_impact_stats[split_key]
        
        if 'active' in stats and 'inactive' in stats:
            # literal < 0 means active branch decision (x > 0)
            if literal < 0:
                impact = stats['active'].get('impact', 0.0)
            else:
                impact = stats['inactive'].get('impact', 0.0)
        else:
             impact = 0.0
             
        literal_impacts.append((literal, impact, layer_name, neuron_id))

    # Sort by impact
    literal_impacts.sort(key=lambda x: x[1]) # Ascending order: lowest impact first
    
    # Determine cutoff index for dropping
    num_literals = len(literal_impacts)
    num_to_drop = int(num_literals * drop_threshold)
    
    dropped_literals_info = literal_impacts[:num_to_drop]
    kept_literals_info = literal_impacts[num_to_drop:]
    
    # Add kept literals to minimized core
    for lit, impact, lname, mid in kept_literals_info:
        minimized_core.append(lit)
        branch_name = "active" if lit < 0 else "inactive"
        logger.debug(f'[MINIMIZE] Keeping literal {lit} ({branch_name}) -> ({lname}, {mid}): impact={impact:.4f}')
        
    # Log dropped
    for lit, impact, lname, mid in dropped_literals_info:
        branch_name = "active" if lit < 0 else "inactive"
        logger.debug(f'[MINIMIZE] Dropping literal {lit} ({branch_name}) -> ({lname}, {mid}): impact={impact:.4f} (Bottom {drop_threshold*100}%)')
    
    dropped_literals = dropped_literals_info

    
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
    # We want to check if the subset of history (negated literals in core) is UNSAT.
    # So we enforce the history decisions (negation of core literals) as unit clauses.
    core_constraints = [[-lit] for lit in minimized_core]
    status = temp_verifier.verify(
        objectives,
        preconditions=core_constraints,
        timeout=time_limit,
        disable_attack=True
    )
    
    # if still UNSAT, return minimized core, else return None
    if status == 'unsat':
        logger.info(f'[MINIMIZE] Minimized core is still UNSAT, returning minimized core')

        # print minimized core
        logger.info(f'[MINIMIZE] Core AFTER minimization (Size {len(minimized_core)}): {minimized_core}')
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
            for condition in incremental_preconditions:
                minimized_core = _minimize_core(objectives, condition, split_impact_stats=verifier.split_impact_stats)

                if minimized_core is not None:
                    incremental_preconditions.remove(condition)
                    incremental_preconditions.append(minimized_core)
                    logger.info(f'[!] Minimized a core from size {len(condition)} to size {len(minimized_core)}')
            
        
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
            

        print(f'{status},{runtime:.04f}')


