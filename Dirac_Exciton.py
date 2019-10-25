import numpy as np
import matplotlib.pyplot as plt
from ase.units import *
from scipy.special import struve,yn
import numba
from scipy.sparse.linalg import eigsh
from itertools import product
from tqdm import tqdm

def lanczos(H,P,niter):
    a=[];b=[];vec=[]
    P/=np.linalg.norm(P)
    vec.append(P)
    a.append(vec[-1].T.conj().dot(H.dot(vec[-1])))
    vec.append(H.dot(vec[-1])-a[-1]*vec[-1])
    vec[-1]/=np.linalg.norm(vec[-1])
    b.append(vec[-2].conj().dot(H.dot(vec[-1])))

    for n in tqdm(range(1,niter)):
        a.append(vec[-1].T.conj().dot(H.dot(vec[-1])))
        b.append(vec[-2].T.conj().dot(H.dot(vec[-1])))
        vec.append(H.dot(vec[-1])-a[-1]*vec[-1]-b[-1]*vec[-2])
        vec[-1]/=np.abs(np.linalg.norm(vec[-1]))
        del vec[0]
        
    return a,b

def keldysh_potential_real(r,epsilon,r0):
    return Hartree*Bohr*np.pi/(2*epsilon*r0)*(struve(0,r/r0)-yn(0,r/r0))

@numba.njit()
def exciton_hamiltonian(HS,E,D,D_conj,WK,x,y):
    NS=HS.shape[0]
    N=E.shape[0]
    for i in range(NS):
        HS[i,i]+=E[x[i],y[i],1]-E[x[i],y[i],0]
        for j in range(NS):
            overlap_c=np.sum(D[x[i],y[i],1].conj()*D[x[j],y[j],1])
            overlap_v=np.sum(D[x[j],y[j],0].conj()*D[x[i],y[i],0])
            HS[i,j]-=WK[(x[i]-x[j])%N,(y[i]-y[j])%N]*overlap_c*overlap_v
    return HS

class Dirac_Exciton(object):
    def __init__(self,params):
        a,t,delta,r0,epsilon=params
        self.a=a
        self.t=t
        self.delta=delta
        self.r0=r0/epsilon
        self.epsilon=epsilon
        self.alpha=1/(a/Bohr*t/Hartree)/epsilon
        
        self.s1=np.array([[0,1],[1,0]])
        self.s2=np.array([[0,-1j],[1j,0]])
        self.s3=np.array([[1,0], [0,-1]])
        self.s0=np.array([[1,0], [0,1]])
        
        self.cell=np.array([[a, 0. ],[-a/2,a*np.sqrt(3)/2]])
        self.icell=2*np.pi*np.linalg.inv(self.cell).T
        
        self.Rydberg=3.3/(epsilon**2)

        
    def dirac_hamiltonian(self,q,tau):
        return self.a*self.t*(tau*q[0]*self.s1+q[1]*self.s2)+self.delta/2*self.s3
    
    def exciton_hamiltonian(self,N,qmax,tau=1):
        self.N=N
        self.qmax=qmax
        self.R=np.zeros((N,N,2))
        self.K=np.zeros((N,N,2))
        self.WR=np.zeros((N,N))
        
        indx=np.array(np.fft.fftfreq(N,1/N),dtype=int)
        for i in range(N):
            for j in range(N):
                self.R[i,j]=self.cell[0]*indx[i]+self.cell[1]*indx[j]
                self.K[i,j]=self.icell[0]*indx[i]/N+self.icell[1]*indx[j]/N
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
        
        self.indexes=[]
        for i in range(N):
            for j in range(N):
                if np.linalg.norm(self.K[i,j])<self.qmax:
                    self.indexes.append((i,j))
        self.NS=len(self.indexes)
        indexes=np.array(self.indexes)
        HS=np.zeros((self.NS,self.NS),dtype=complex)
        self.HS=exciton_hamiltonian(HS,self.E,self.D,self.D.conj(),self.WK,indexes[:,0],indexes[:,1])
        
        self.P=np.zeros((N,N),dtype=complex)
        for i in range(N):
            for j in range(N):
                self.P[i,j]=np.dot(self.D[i,j,:,1].T.conj(),np.dot(self.s1+1j*self.s2,self.D[i,j,:,0]))
        self.P*=self.a*self.t*tau/Hartree/Bohr
        self.PS=np.array([self.P[i,j] for i,j in self.indexes])
    
    def exciton_states(self,nstates,plot=True,save=True,data=False):
        if hasattr(self,'HS'):
            pass
        else:
            print('Please first construct Exciton Hamiltonian')
            return None
        number_of_states=4
        ES,DS=eigsh(self.HS,which='SA',k=number_of_states)
        if plot:
            for state in range(number_of_states):
                if ES[state]<self.delta:
                    plt.figure(figsize=(4,4))
                    wave=np.zeros((self.N,self.N))
                    for i,index in enumerate(self.indexes):
                        x,y=index
                        wave[x,y]=np.abs(DS[i,state])
                    plt.contourf(np.fft.fftshift(self.K[:,:,0]),
                                 np.fft.fftshift(self.K[:,:,1]),
                                 np.fft.fftshift(wave),100,cmap='gnuplot')
                    plt.axis('equal')
                    plt.xlim([-self.qmax,self.qmax])
                    plt.ylim([-self.qmax,self.qmax])
                    plt.title('Binding enerdy {:.{prec}f} eV'.format(self.delta-ES[state],prec=3))
                    plt.xlabel('k$_x$ [$\AA^{-1}$]')
                    plt.ylabel('k$_y$ [$\AA^{-1}$]')
                    if save:
                        plt.savefig('figures/'+str(state)+'.png',dpi=600)
        if data: return ES,DS
    
    
    def susceptibility(self,wmin=0,wmax=2,eta=0.01,npoints=1001,plot=True,exciton=True):
        
        if hasattr(self,'HS'):
            pass
        else:
            print('Please first construct Exciton Hamiltonian')
            return None
        
        w=np.linspace(wmin,wmax,npoints)
        self.E/=Hartree;w/=Hartree;eta/=Hartree
        
        
        self.chi0=np.zeros(npoints,dtype=complex)
        for i in range(npoints):
            self.chi0[i]+=2*np.sum(np.abs(self.P)**2/(w[i]+1j*eta-(self.E[:,:,1]-self.E[:,:,0])))
            self.chi0[i]-=2*np.sum(np.abs(self.P)**2/(w[i]+1j*eta+(self.E[:,:,1]-self.E[:,:,0])))
        self.chi0/=(self.N**2)
        
        if exciton:
            self.chi=np.zeros(npoints,dtype=complex)
            self.ES,self.DS=np.linalg.eigh(self.HS/Hartree)
            self.OS=np.abs(np.sum(self.DS*self.PS[:,None],axis=0))**2 
            for i in range(self.NS):
                self.chi+=2*self.OS[i]/(w-self.ES[i]+1j*eta)
                self.chi-=2*self.OS[i]/(w+self.ES[i]+1j*eta)
            self.chi/=self.NS   
        
        self.E*=Hartree;w*=Hartree;eta*=Hartree
        
        self.w=w
        if plot:
            
            if exciton:
                plt.plot(w,self.chi.real,label='Re[$\chi(\omega)$]')
                plt.plot(w,self.chi.imag,label='Im[$\chi(\omega)$]')
                
            plt.plot(w,self.chi0.real,label='Re[$\chi_0(\omega)$]')
            plt.plot(w,self.chi0.imag,label='Im[$\chi_0(\omega)$]')
            
      
            
            plt.grid()
            plt.xlim([0,2])
            plt.legend()
            plt.xlabel('$\omega$ [eV]')
            plt.ylabel('$\chi$ [$e^2 a^2_B / E_h$]')
            plt.savefig('figures/susceptibility.png',dpi=600)

        

        
