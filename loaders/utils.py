import numpy as np
import scipy.io

def load_mat_as_dict(the_path: str):
    """Loads a mat file as a dictionary. 

    Arguments:
        the_path: a string containing the path of a mat file.

    Returns:
        The mat file, formatted as a dict instead of a weird array.
    """
    mat = scipy.io.loadmat(the_path)
    for key, val in mat.items():
        mat[key] = _clean_up(val)
    return mat

def _clean_up(mat):
    """Cleans up an object numpy array recursively.

    Arguments:
        mat: a mat file

    Returns:
        A cleaned up dictionary.
    """
    if isinstance(mat, np.ndarray):
        if len(mat.dtype) > 1:
            # This needs to be cleaned up.
            dict_mat = {}
            for name in mat.dtype.names:
                root = mat[name]
                while len(root) == 1 and root.dtype == np.dtype('O'):
                    root = root[0]
                dict_mat[name] = _clean_up(root)
            return dict_mat
        elif mat.dtype == np.dtype('O'):
            list_mat = []
            for _, item in enumerate(mat):
                root = item
                while len(root) == 1 and root.dtype == np.dtype('O'):
                    root = root[0]

                list_mat.append(_clean_up(root))
            return list_mat
        mat = mat.squeeze()
        if mat.shape == tuple():
            if np.issubdtype(mat.dtype, np.integer):
                mat = int(mat)
            elif np.issubdtype(mat.dtype, np.floating):
                mat = float(mat)
            else:
                mat = str(mat)
        return mat
    return mat

if __name__ == '__main__':
    the_mat = load_mat_as_dict('crcns-ringach-data/neurodata/ac1/ac1_u004_000.mat')