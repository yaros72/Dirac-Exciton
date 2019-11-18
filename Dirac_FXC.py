import numpy as np
import matplotlib.pyplot as plt
from ase.units import *
from scipy.special import struve,yn
import numba
from scipy.sparse.linalg import eigsh
from itertools import product
from tqdm import tqdm


def keldysh_potential_real(r,epsilon,r0):
    return np.pi/(2*epsilon*r0)*(struve(0,r/r0)-yn(0,r/r0))

@numba.njit()
def exciton_hamiltonian(HS,E,D,D_conj,WK,indexes):
    N=E.shape[0]
    NS=HS.shape[0]
    for i in range(NS):
        x1,y1,c1,v1=indexes[i]
        HS[i,i]+=E[x1,y1,c1]-E[x1,y1,v1]
        for j in range(NS):
            x2,y2,c2,v2=indexes[j]
            overlap_c=np.sum(D[x1,y1,c1].conj()*D[x2,y2,c2])
            overlap_v=np.sum(D[x2,y2,v2].conj()*D[x1,y1,v1])
            HS[i,j]-=WK[(x1-x2)%N,(y1-y2)%N]*overlap_c*overlap_v
    return HS

@numba.njit()
def lindhard(chi,E,D,F):
    N=E.shape[0];norb=E.shape[2]
    for x1 in range(N):
        for y1 in range(N):
            for x2 in range(N):
                for y2 in range(N):
                    for a in range(norb):
                        for b in range(norb):
                            if (F[x1,y1,a]-F[x2,y2,b])!=0:
                                rho=np.abs(np.sum(D[x1,y1,:,a].conj()*D[x2,y2,:,b]))**2
                                factor=(F[x1,y1,a]-F[x2,y2,b])/(E[x1,y1,a]-E[x2,y2,b])
                                chi[(x1-x2)%N,(y1-y2)%N]+=rho*factor
    return chi

class Dirac_FXC(object):
    def __init__(self,params):
        a,t,delta,r0,epsilon=params
        
        a/=Bohr
        r0/=Bohr
        t/=Hartree
        delta/=Hartree
        
        self.a=a
        self.t=t
        self.delta=delta
        self.r0=r0
        self.epsilon=epsilon
        self.alpha=1/(a*t)/epsilon
        
        self.s1=np.array([[0,1],[1,0]])
        self.s2=np.array([[0,-1j],[1j,0]])
        self.s3=np.array([[1,0], [0,-1]])
        self.s0=np.array([[1,0], [0,1]])
        
        self.cell=np.array([[a, 0. ],[-a/2,a*np.sqrt(3)/2]])
        self.icell=2*np.pi*np.linalg.inv(self.cell).T
        
        
    def dirac_hamiltonian(self,q,tau):
        return self.a*self.t*(tau*q[0]*self.s1+q[1]*self.s2)+self.delta*self.s3
    
    def exciton_hamiltonian(self,N,tau=1,exciton=False):
        self.N=N
        self.R=np.zeros((N,N,2))
        self.K=np.zeros((N,N,2))
        self.WR=np.zeros((N,N))
        self.VK=np.zeros((N,N))
        
        indx=np.array(np.fft.fftfreq(N,1/N),dtype=int)
        for i in range(N):
            for j in range(N):
                self.R[i,j]=self.cell[0]*indx[i]+self.cell[1]*indx[j]
                self.K[i,j]=self.icell[0]*indx[i]/N+self.icell[1]*indx[j]/N
                self.VK[i,j]=2*np.pi/np.linalg.norm(self.K[i,j])
                self.WR[i,j]=keldysh_potential_real(np.linalg.norm(self.R[i,j]),self.epsilon,self.r0) 
        self.WR[0,0]=0
        corrections=10
        for i in range(corrections):  
            self.WK=np.abs(np.fft.fft2(self.WR))/N/N
            self.WR[0,0]=np.sum(self.WK)
        
        self.E=np.zeros((N,N,2))
        self.D=np.zeros((N,N,2,2),dtype=complex)
        H=np.zeros((N,N,2,2),dtype=complex)
        for i in range(N):
            for j in range(N):
                H[i,j]=self.dirac_hamiltonian(self.K[i,j],tau)
                e,d=np.linalg.eigh(H[i,j])
                self.E[i,j]=e
                self.D[i,j]=d
        self.F=np.array(self.E<0,dtype=int)/self.N/self.N
        
        self.indexes=[]
        for i in range(N):
            for j in range(N):
                self.indexes.append((i,j,1,0))
        self.NS=len(self.indexes)
        
        
        indexes=np.array(self.indexes)
        HS=np.zeros((self.NS,self.NS),dtype=complex)
        self.HS=exciton_hamiltonian(HS,self.E,self.D,self.D.conj(),self.WK,indexes)
        
        self.P=np.zeros((N,N,2,2),dtype=complex)
        for i in range(N):
            for j in range(N):
                for n in range(2):
                    for m in range(2):
                        self.P[i,j,n,m]=self.a*self.t*np.dot(self.D[i,j,:,n].T.conj(),np.dot(self.s1,self.D[i,j,:,m]))
        self.PS=np.array([self.P[i,j,c,v] for i,j,c,v in self.indexes])
    
    def susceptibility_zero(self):
        ES,DS=np.linalg.eigh(self.HS)
        OS=np.abs(np.sum(DS*self.PS[:,None],axis=0))**2 
      
        self.chi0=0
        for i in range(self.N):
            for j in range(self.N):
                for n in range(2):
                    for m in range(2):
                        if (self.F[i,j,n]-self.F[i,j,m])!=0:
                            self.chi0+=self.P[i,j,n,m]*self.P[i,j,m,n]*(self.F[i,j,n]-self.F[i,j,m])/(self.E[i,j,n]-self.E[i,j,m])              
        self.chi=-4*np.sum(OS/ES)/self.NS
        self.fxc=1/self.chi0-1/self.chi
        
        self.ES=ES
        self.DS=DS
        self.OS=OS
    
    def polarization(self,N,tau):
        polar=np.zeros((N,N),dtype=complex)
        K=np.zeros((N,N,2))
        V=np.zeros((N,N))
        indx=np.array(np.fft.fftfreq(N,1/N),dtype=int)
        for i in range(N):
            for j in range(N):
                K[i,j]=self.icell[0]*indx[i]/N+self.icell[1]*indx[j]/N
                V[i,j]=2*np.pi/np.linalg.norm(K[i,j])
        E=np.zeros((N,N,2))
        D=np.zeros((N,N,2,2),dtype=complex)
        for i in range(N):
            for j in range(N):
                e,d=np.linalg.eigh(self.dirac_hamiltonian(K[i,j],tau))
                E[i,j]=e;D[i,j]=d
        F=np.array(E<0,dtype=int)/N/N
        self.polar=lindhard(polar,E,D,F)/(2*np.pi)**2
        self.epsilon=1-V*self.polar
        self.V=V
        self.K=np.fft.fftshift(K,axes=(0,1)).transpose([2,0,1])
        self.E=E