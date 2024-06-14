import os

data_dir = "data"

def find_data_dir(current_dir='.', max_dirs_up=4):
    """Looks for the data root directory.

    Args:
        current_dir (str): The relative or absolute path of the directory we look in.
        max_dirs_up (int): Number of dirs up the dir tree the function should look for 'data_dir'

    Returns:
        string: Absolute path to the data directory.

    Notes:
        The function will check for 'data_dir' recursively in the current dir
        and then 'max_dirs_up' dirs up the dir tree.
    """
    current_dir = os.path.join(os.getcwd(), current_dir)

    # recursive search in current dir
    for (root,dirs,_) in os.walk(current_dir):
        for d in dirs:
            if os.path.basename(d) == data_dir:
                return os.path.join(root, d)

    # max_dirs_up search up the directory tree
    for _ in range(max_dirs_up):
        os.path.join(current_dir,'..')
        dirs = [d for d in os.listdir(current_dir) if os.path.join(current_dir,d)]
        for d in dirs:
            if os.path.basename(d) == data_dir:
                return os.path.join(current_dir, d)

    raise Exception("No data directory was found")