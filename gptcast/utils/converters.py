import numpy as np

def dbz_to_rainfall(dBZ, a=200., b=1.6, cutoff=0.04):
    rr = np.power(10, (dBZ - 10 * np.log10(a)) / (10 * b))
    rr[rr < cutoff] = 0
    return rr


def rainfall_to_dbz(rainfall, a=200., b=1.6, cutoff=0.):
    dbz = 10 * np.log10(a) + 10 * b * np.log10(rainfall)
    dbz[dbz < cutoff] = 0
    return dbz