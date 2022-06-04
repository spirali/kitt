import multiprocessing
from typing import Union


def map_items(fn, arguments, parallel: Union[int, bool] = True, use_tqdm=True):
    """
    Maps `fn` over `arguments` in a serial or parallel fashion with a unified interface.
    If `parallel` is False, the mapping will be done in serial.
    If `parallel` is True, the mapping will be done over all available cores in processes.
    If `parallel` is a positive number, the mapping will be done over `parallel` cores in
    processes.
    """

    def compute_iterator(iterator):
        if use_tqdm:
            import tqdm

            iterator = tqdm.tqdm(iterator, total=len(arguments))
        return list(iterator)

    if parallel:
        n_procs = multiprocessing.cpu_count() if parallel is True else parallel
        with multiprocessing.Pool(n_procs) as pool:
            iterator = pool.imap(fn, arguments)
            return compute_iterator(iterator)
    else:
        iterator = map(fn, arguments)
        return compute_iterator(iterator)
