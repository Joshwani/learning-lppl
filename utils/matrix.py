# def _yi():
#     return [np.log(p) for p in DataSeries[1]]

# def _fi(tc_tau, m):
#     return np.power(tc_tau, m)

# def _gi(tc_tau, m, w):
#     return np.multiply(
#         _fi(tc_tau, m), 
#         np.cos(np.multiply([w] * len(tc_tau), np.log(tc_tau)))
#     )

# def _hi(tc_tau, m, w):
#     return np.multiply(
#         _fi(tc_tau, m), 
#         np.sin(np.multiply([w] * len(tc_tau), np.log(tc_tau)))
#     )

# def _fi_pow_2(tc_tau, m):
#     return np.power(_fi(tc_tau, m), 2)

# def _gi_pow_2(tc_tau, m, w):
#     return np.power(_gi(tc_tau, m, w), 2)

# def _hi_pow_2(tc_tau, m, w):
#     return np.power(_hi(tc_tau, m, w), 2)

# def _figi(tc_tau, m, w):
#     return np.multiply(_fi(tc_tau, m), _gi(tc_tau, m, w))

# def _fihi(tc_tau, m, w):
#     return np.multiply(_fi(tc_tau, m), _hi(tc_tau, m, w))

# def _gihi(tc_tau, m, w):
#     return np.multiply(_gi(tc_tau, m, w), _hi(tc_tau, m, w))

# def _yifi(tc_tau, m):
#     return np.multiply(_yi(), _fi(tc_tau, m))

# def _yigi(tc_tau, m, w):
#     return np.multiply(_yi(), _gi(tc_tau, m, w))

# def _yihi(tc_tau, m, w):
#     return np.multiply(_yi(), _hi(tc_tau, m, w))

def _yi():
    return [np.log(p) for p in DataSeries[1]]

def _fi(tc, m):
    return [np.power((tc - t), m) for t in DataSeries[0]]

def _gi(tc, m, w):
    return [np.power((tc - t), m) * np.cos(w * np.log(tc - t)) for t in DataSeries[0]]

def _hi(tc, m, w):
    return [np.power((tc - t), m) * np.sin(w * np.log(tc - t)) for t in DataSeries[0]]

def _fi_pow_2(tc, m):
    return np.power(_fi(tc, m), 2)

def _gi_pow_2(tc, m, w):
    return np.power(_gi(tc, m, w), 2)

def _hi_pow_2(tc, m, w):
    return np.power(_hi(tc, m, w), 2)

def _figi(tc, m, w):
    return np.multiply(_fi(tc, m), _gi(tc, m, w))

def _fihi(tc, m, w):
    return np.multiply(_fi(tc, m), _hi(tc, m, w))

def _gihi(tc, m, w):
    return np.multiply(_gi(tc, m, w), _hi(tc, m, w))

def _yifi(tc, m):
    return np.multiply(_yi(), _fi(tc, m))

def _yigi(tc, m, w):
    return np.multiply(_yi(), _gi(tc, m, w))

def _yihi(tc, m, w):
    return np.multiply(_yi(), _hi(tc, m, w))