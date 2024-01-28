if __name__ == "__main__":
    import os
    from mylib.datamakers.join import main

    path_in = 'data-2/augmented'
    names = ['replace-test', 'replace-test2']
    path_out = 'data-2/augmented'

    main(
        files_in=[os.path.join(path_in, f'{name}.parquet') for name in names],
        names=names,
        path_out=path_out,
        name_out='joined-test'
    )