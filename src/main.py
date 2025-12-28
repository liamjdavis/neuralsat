import argparse
import torch
import time
import os

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

    for v in temp_verifier.all_conflict_clauses.values():
        new_cores.extend(v)

    logger.info(f'[!] Re-solving found {len(new_cores)} new UNSAT cores.')
    
    # clean up verifier
    del temp_verifier
    if 'cuda' in args.device:
        gc.collect()
        torch.cuda.empty_cache()

    return new_cores

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
            new_preconditions.extend(v)
        incremental_preconditions.extend(new_preconditions)
        verifier.all_conflict_clauses = {} # clear for next run
        logger.info(f'[!] Transferred {len(new_preconditions)} UNSAT cores')
        
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

        # if condition is SAT, reverify under time limit
        if status == 'sat':
            # Reverify using BaB with time limit
            logger.debug(f'[DEBUG] Calling _resolve_with_bab with args.resolve_time_limit={args.resolve_time_limit}')
            new_cores = _resolve_with_bab(objectives=objectives, preconditions=incremental_preconditions, time_limit=args.resolve_time_limit)

            # Add new cores to preconditions
            incremental_preconditions.extend(new_cores)

        print(f'{status},{runtime:.04f}')


