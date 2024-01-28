if __name__ == "__main__":
    import os
    from mylib.datamakers.collect_chunks import main

    path_in = 'data-2/augmented'
    names = ['replace-test', 'replace-test2']
    path_out = path_in

    for name in names:
        main(os.path.join(path_in, name), path_out, name)
