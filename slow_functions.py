import numpy as np
from tqdm import tqdm
def lanczos(H,P,n_iter):
    a=[];b=[];vec=[]
    P/=np.linalg.norm(P)
    vec.append(P)
    a.append(vec[-1].T.conj().dot(H.dot(vec[-1])))
    vec.append(H.dot(vec[-1])-a[-1]*vec[-1])
    vec[-1]/=np.linalg.norm(vec[-1])
    b.append(vec[-2].conj().dot(H.dot(vec[-1])))

    for n in tqdm(range(1,n_iter)):
        a.append(vec[-1].T.conj().dot(H.dot(vec[-1])))
        b.append(vec[-2].T.conj().dot(H.dot(vec[-1])))
        vec.append(H.dot(vec[-1])-a[-1]*vec[-1]-b[-1]*vec[-2])
        vec[-1]/=np.abs(np.linalg.norm(vec[-1]))
        del vec[0]
        
    return a,b