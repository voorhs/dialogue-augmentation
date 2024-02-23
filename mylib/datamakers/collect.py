import os
from datasets import disable_caching
from mylib.datamakers.utils import collect_chunks


def main(path_in, path_out):
    disable_caching()

    collect_chunks(
        path_in=path_in,
        path_out=path_out,
        as_string=False
    )
