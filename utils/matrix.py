def _yi():
    return [np.log(p) for p in DataSeries[1]]

def _fi(tc_tau, m):
    return np.power(tc_tau, m)

def _gi(tc_tau, m, w):
    return np.multiply(
        _fi(tc_tau, m), 
        np.cos(np.multiply([w] * len(tc_tau), np.log(tc_tau)))
    )

def _hi(tc_tau, m, w):
    return np.multiply(
        _fi(tc_tau, m), 
        np.sin(np.multiply([w] * len(tc_tau), np.log(tc_tau)))
    )

def _fi_pow_2(tc_tau, m):
    return np.power(_fi(tc_tau, m), 2)

def _gi_pow_2(tc_tau, m, w):
    return np.power(_gi(tc_tau, m, w), 2)

def _hi_pow_2(tc_tau, m, w):
    return np.power(_hi(tc_tau, m, w), 2)

def _figi(tc_tau, m, w):
    return np.multiply(_fi(tc_tau, m), _gi(tc_tau, m, w))

def _fihi(tc_tau, m, w):
    return np.multiply(_fi(tc_tau, m), _hi(tc_tau, m, w))

def _gihi(tc_tau, m, w):
    return np.multiply(_gi(tc_tau, m, w), _hi(tc_tau, m, w))

def _yifi(tc_tau, m):
    return np.multiply(_yi(), _fi(tc_tau, m))

def _yigi(tc_tau, m, w):
    return np.multiply(_yi(), _gi(tc_tau, m, w))

def _yihi(tc_tau, m, w):
    return np.multiply(_yi(), _hi(tc_tau, m, w))