import numpy as np
import sys

sys.path.append("../")

from loaders import pvc1


def get_max_r2(Yall_st):
    # From Sahani & Linden (2003), How Linear are Auditory Cortical Responses?
    assert Yall_st.shape[0] < Yall_st.shape[1]
    # Equation 1
    signal_power = (
        1
        / (Yall_st.shape[0] - 1)
        * (Yall_st.shape[0] * Yall_st.mean(0).var() - Yall_st.var(1).mean())
    )
    response_power = Yall_st.mean(0).var()
    max_r2 = signal_power / response_power
    return max_r2


if __name__ == "__main__":
    """Only for pvc1. See generate_hyperflow for the method for HyperFlow."""
    maxr2s = []
    for single_cell in range(23):
        loader = pvc1.PVC1(
            "/mnt/e/data_derived/crcns-pvc1/",
            nt=1,
            ntau=10,
            nframedelay=0,
            repeats=True,
            single_cell=single_cell,
            split="report",
        )
        Ys = []
        for _, _, Y, _ in loader:
            Ys.append(Y)

        Y = np.concatenate(Ys, axis=0)

        maxr2 = get_max_r2(Y.T)

        print(single_cell, maxr2)
        maxr2s.append(maxr2)

    print(maxr2s)