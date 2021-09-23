#This program read filprojcplx and bands.dat.gnu and output files for origin to plot arpes spectrum
#IMPORTANT!!!!!!!!!!!!!!!!!!!!
#This program assume sample to be periodic in x,y direction, only consider contributions of 2p orbitals and do
#not consider any decay in z directions! Namely we are only calculating <e^ik\dot r|r|p>
#Jingwei Jiang UC Berkeley 7/3/2019
#ver 3.0
#new in version2.0: parallelization implemented
#new in version3.0: bug fixed, code accelerated considerably.
import numpy as np
import sys
import scipy
import scipy.integrate as integrate
import scipy.special as special
import math
import cmath
from mpi4py import MPI
#import mkl
#mkl.set_num_threads(1)

#definition of atomic orbitals and coordinate transformation from cartesian to spherical.
def px(r,th,ph):
        px1=math.sqrt(3)*math.sin(th)*math.cos(ph)*math.sqrt(1/(4*math.pi))/(2*math.sqrt(6))*r*math.exp(-r/2)
        return px1


def py(r,th,ph):
        py1=math.sqrt(3)*math.sin(th)*math.sin(ph)*math.sqrt(1/(4*math.pi))/(2*math.sqrt(6))*r*math.exp(-r/2)
        return py1


def pz(r,th,ph):
        pz1=math.sqrt(3)*math.cos(th)*math.sqrt(1/(4*math.pi))/(2*math.sqrt(6))*r*math.exp(-r/2)
        return pz1


def x(r,th,ph):
        x1=r*math.sin(th)*math.cos(ph)
        return x1


def y(r,th,ph):
        y1=r*math.sin(th)*math.sin(ph)
        return y1


def z(r,th,ph):
        z1=r*math.cos(th)
        return z1


def s(r,th,ph):
        s=2*math.exp(-r)*math.sqrt(1/(4*math.pi))
        return s


def splane(r,th,ph,kf):
        kr=np.linalg.norm(kf)
        sp=special.spherical_jn(0,kr*r)*special.sph_harm(0,0,ph,th)
        return sp

def pplane(r,th,ph,kf):
        if abs(kf[0])>0.0001:
                kph=math.atan(kf[1]/kf[0])
        elif abs(kf[1])>0.0001:
                kph=math.acot(kf[0]/kf[1])
        else:
                kph=0
        kr=np.linalg.norm(kf)
        kth=math.acos(kf[2]/kr)
        pp=1j*special.spherical_jn(1,kr*r)*special.sph_harm(1,1,ph,th)*special.sph_harm(1,1,kph,kth).conjugate()+\
        1j*special.spherical_jn(1,kr*r)*special.sph_harm(0,1,ph,th)*special.sph_harm(0,1,kph,kth).conjugate()+\
        1j*special.spherical_jn(1,kr*r)*special.sph_harm(-1,1,ph,th)*special.sph_harm(-1,1,kph,kth).conjugate()
        return pp

def dplane(r,th,ph,kf):
        if abs(kf[0])>0.0001:
                kph=math.atan(kf[1]/kf[0])
        elif abs(kf[1])>0.0001:
                kph=math.acot(kf[0]/kf[1])
        else:
                kph=0
        kr=np.linalg.norm(kf)
        kth=math.acos(kf[2]/kr)
        dp=-special.spherical_jn(2,kr*r)*(special.sph_harm(-2,2,ph,th)*special.sph_harm(-2,2,kph,kth).conjugate()+special.sph_harm(-1,2,ph,th)*special.sph_harm(-1,2,kph,kth).conjugate()+special.sph_harm(0,2,ph,th)*special.sph_harm(0,2,kph,kth).conjugate()+special.sph_harm(1,2,ph,th)*special.sph_harm(1,2,kph,kth).conjugate()+special.sph_harm(2,2,ph,th)*special.sph_harm(2,2,kph,kth).conjugate())
        return dp
#this function calculate transition dipole matrix element of orbital type orb, polarization direction pol and plane wave
def tran(orb,pol,kf):
        if abs(kf[0])>0.0001:
                phk=math.atan(kf[1]/kf[0])
        elif abs(kf[1])>0.0001:
                phk=math.acot(kf[0]/kf[1])
        else:
                phk=math.pi/3
        kr=np.linalg.norm(kf)
        thk=math.acos(kf[2]/kr)
        if 'px' in orb:
                pxre=integrate.quad(lambda r:special.spherical_jn(0,kr*r)*pol[0]*r**4*math.exp(-r/2),0,np.inf)[0]*0.11785113\
                +pol[1]*0.064549722*integrate.quad(lambda r:1j*special.spherical_jn(2,kr*r)*r**4*math.exp(-r/2)*(special.sph_harm(2,2,phk,thk)-special.sph_harm(-2,2,phk,thk)),0,np.inf)[0]\
                -pol[0]*0.064549722*integrate.quad(lambda r:special.spherical_jn(2,kr*r)*r**4*math.exp(-r/2)*(special.sph_harm(2,2,phk,thk)+special.sph_harm(-2,2,phk,thk)),0,np.inf)[0]+pol[0]*0.052704628*integrate.quad(lambda r:special.spherical_jn(2,kr*r)*r**4*math.exp(-r/2)*special.sph_harm(0,2,phk,thk),0,np.inf)[0]\
                +pol[2]*0.064549722*integrate.quad(lambda r:special.spherical_jn(2,kr*r)*r**4*math.exp(-r/2)*(special.sph_harm(1,2,phk,thk)-special.sph_harm(-1,2,phk,thk)),0,np.inf)[0]
#               print('pxre:',pxre)
                return pxre
        elif 'py' in orb:
                pyre=integrate.quad(lambda r:special.spherical_jn(0,kr*r)*pol[1]*r**4*math.exp(-r/2),0,np.inf)[0]*0.11785113\
                +pol[0]*0.064549722*integrate.quad(lambda r:1j*special.spherical_jn(2,kr*r)*r**4*math.exp(-r/2)*(special.sph_harm(2,2,phk,thk)-special.sph_harm(-2,2,phk,thk)),0,np.inf)[0]\
                +pol[1]*0.064549722*integrate.quad(lambda r:special.spherical_jn(2,kr*r)*r**4*math.exp(-r/2)*(special.sph_harm(2,2,phk,thk)+special.sph_harm(-2,2,phk,thk)),0,np.inf)[0]-pol[1]*0.052704628*integrate.quad(lambda r:special.spherical_jn(2,kr*r)*r**4*math.exp(-r/2)*special.sph_harm(0,2,phk,thk),0,np.inf)[0]\
                -pol[2]*0.064549722*integrate.quad(lambda r:1j*special.spherical_jn(2,kr*r)*r**4*math.exp(-r/2)*(special.sph_harm(1,2,phk,thk)+special.sph_harm(-1,2,phk,thk)),0,np.inf)[0]
#               print('pyre:',pyre)
                return pyre
        elif 'pz' in orb:
                pzre=integrate.quad(lambda r:special.spherical_jn(0,kr*r)*pol[2]*r**4*math.exp(-r/2),0,np.inf)[0]*0.11785113\
                +pol[0]*0.064549722*integrate.quad(lambda r:special.spherical_jn(2,kr*r)*r**4*math.exp(-r/2)*(special.sph_harm(1,2,phk,thk)-special.sph_harm(-1,2,phk,thk)),0,np.inf)[0]\
                -pol[1]*0.064549722*integrate.quad(lambda r:1j*special.spherical_jn(2,kr*r)*r**4*math.exp(-r/2)*(special.sph_harm(1,2,phk,thk)+special.sph_harm(-1,2,phk,thk)),0,np.inf)[0]\
                -pol[2]*0.105409255*integrate.quad(lambda r:special.spherical_jn(2,kr*r)*r**4*math.exp(-r/2)*special.sph_harm(0,2,phk,thk),0,np.inf)[0]
#               print('pzre:',pzre)
                return pzre
        elif 's' in orb:
                sre=pol[0]*(1j)*0.816496581*integrate.quad(lambda r:special.spherical_jn(1,kr*r)*r**3*math.exp(-r)*(special.sph_harm(1,1,phk,thk)-special.sph_harm(-1,1,phk,thk)),0,np.inf)[0]\
                +pol[1]*(1j)*0.816496581*integrate.quad(lambda r:scipy.imag(special.spherical_jn(1,kr*r)*r**3*math.exp(-r)*(special.sph_harm(1,1,phk,thk)+special.sph_harm(-1,1,phk,thk))),0,np.inf)[0]\
                -pol[2]*(1j)*1.154700538*integrate.quad(lambda r:special.spherical_jn(1,kr*r)*r**3*math.exp(-r)*special.sph_harm(0,1,phk,thk),0,np.inf)[0]
                return sre
        else:
                print('no p/s contribution...')
                exit(1)
#       elif 's' in orb:
#               sre=1j*integrate.tplquad(lambda r,th,ph:scipy.imag(s(r,th,ph)*(pol[0]*x(r,th,ph)+pol[1]*y(r,th,ph)+pol[2]*z(r,th,ph))*pplane(r,th,ph,kf).conjugate()*r**2*math.sin(th)),0,2*math.pi,lambda ph:0,lambda ph:math.pi,lambda th,ph:0,lambda th,ph:1000)[0]
#               return sre
#this function calculate the overlap of atomic orbitals and plane waves


def S(orb,kf):
        if abs(kf[0])>0.0001:
                phk=math.atan(kf[1]/kf[0])
        elif abs(kf[1])>0.0001:
                phk=math.acot(kf[0]/kf[1])
        else:
                phk=math.pi/3
        kr=np.linalg.norm(kf)
        thk=math.acos(kf[2]/kr)
        if 'pz' in orb:
                pzse=-1j*0.204124145*integrate.quad(lambda r:special.spherical_jn(1,kr*r)*special.sph_harm(0,1,phk,thk)*r**3*math.exp(-r/2),0,np.inf)[0]
#               print('pzse:',pzse)
                return pzse
        elif 'px' in orb:
                pxse=1j*0.144337567*integrate.quad(lambda r:special.spherical_jn(1,kr*r)*(special.sph_harm(1,1,phk,thk)-special.sph_harm(-1,1,phk,thk))*r**3*math.exp(-r/2),0,np.inf)[0]
#               print('pxse:',pxse)
                return pxse
        elif 'py' in orb:
                pyse=1j*0.144337567*integrate.quad(lambda r:scipy.imag(special.spherical_jn(1,kr*r)*(special.sph_harm(1,1,phk,thk)+special.sph_harm(-1,1,phk,thk))*r**3*math.exp(-r/2)),0,np.inf)[0]
#               print('pyse:',pyse)
                return pyse
        elif 's' in orb:
                sse=2*integrate.quad(lambda r:special.spherical_jn(0,kr*r)*math.exp(-r)*r**2,0,np.inf)[0]
                return sse
        else:
                print('no p/s contribution...')
                exit(1)

#main program
#potential input parameters: alat,pol,eph,evac,inputfile name of pw.x
if __name__=="__main__":
#alat of bands calculation
        comm=MPI.COMM_WORLD
        rank=comm.Get_rank()
        size=comm.Get_size()
        alat=23.8011
        if rank==0:
                print('alat=',alat)
        infil=open('filprojcplx','r').readlines()
        #polarisation direction
        pol=np.array([0,0,1])
        #find the starting line to read pdos information
        for i in range(7,len(infil)):
                if len(infil[i].split())==7:
                        istart=i
                        break
        #number of states
        nst=int(infil[istart-2].split()[0])
        #number of atoms
        natm=int(infil[1].split()[6])
        #number of k points
        nk=int(infil[istart-2].split()[1])
        #number of bands
        nbnd=int(infil[istart-2].split()[2])
        if rank ==0:
                print('number of states are:',nst)
                print('number of k points are:',nk)
                print('number of bands are:',nbnd)
        #array that stores bandstructure data
        kbarp=np.zeros((nk,nbnd,2),dtype=complex)
        klist=np.zeros(nk)
        #photon energy eV
        eph=103.19
        #vacuum level(eV) not used in ver0.0
        evac=0.075
        #normalise pol
        pol=pol/np.linalg.norm(pol)
        if rank ==0:
                print('polarisation direction:',pol)
        #get bandstructures
        inbfil=open('bands.dat.gnu','r').readlines()
        for i in range(nk):
                klist[i]=float(inbfil[i].split()[0])
        for i in range(nk):
                for j in range(nbnd):
                        kbarp[i,j,0]=float(inbfil[i+j*(nk+1)].split()[1])
        datfil=open('bands.dat','r').readlines()
        l=0
        kp=np.zeros((nk,3))
        for i in range(len(datfil)):
                line=datfil[i].split()
                if len(line) == 3:
                        kp[l]=[float(kk) for kk in line]
                        l+=1
        kp=kp/alat*2*math.pi
        posfil=open('newinag','r').readlines()
        if rank ==0:
                print('read atomic positions in bandsin......')
        l=0
        pos=np.zeros((natm,3))
        for i in range(len(posfil)):
                if 'atomic_positions angstrom' in posfil[i].lower():
                        for j in range(natm):
                                pos[l]=[float(pp) for pp in posfil[i+j+1].split()[1:4]]
                                l+=1
        if l==0:
                if rank==0:
                        print('no atomic positions found in bandsin, exiting...')
                        sys.stdout.flush()
                exit(1)
        else:
                if rank==0:
                        print('atomic positions found.')
                sys.stdout.flush()
        #convert angstrom to bohr
        pos=pos/0.529177
        #get separate jobs for each mpi tasks
        if size<=nbnd:
                if rank==0:
                        print('class one parallelization: ntasks<nbnd')
                        print('parallel over bands')
                if nbnd - size * int(nbnd/size) == 0:
                        my_lenb = int(nbnd/size)
                        ave_lenb = my_lenb # ave_len is just my_len except for the last processor
                        lastrank=size-1
                else:
                        if rank==0:
                                print("Warnings:# of MPI tasks are not optimal, number of bands are:",nbnd)
                        sys.stdout.flush()
                        my_lenb = int(nbnd/size)+1
                        lastrank=int(nbnd/my_lenb)
                        ave_lenb = my_lenb
                        if rank == lastrank: # the last processor
                                my_lenb = nbnd - ave_lenb*lastrank
                if rank==0:
                        print('rank 0 has',ave_lenb,'bands.')
                if rank==lastrank:
                        print('last rank has',my_lenb,'bands.')
                my_startk=0
                my_endk=my_startk+nk
                my_startb=[int(ele) for ele in (np.zeros(my_endk)+rank*ave_lenb)]
                my_endb=[int(ele)for ele in (np.zeros(my_endk)+my_startb+my_lenb)]
                print('rank: ',rank,'has',nk,'k-points and ',my_startb,' to ',my_endb,'bands')
                print('lastrank is:',lastrank)
                sys.stdout.flush()
        elif size<=nk*nbnd:
                if rank==0:
                        print('class two parallelization: ntasks<nk*nbnd')
                        print('parallel over bands and kpoints')
                #get my_lentot
                if nk*nbnd-size*int(nk*nbnd/size)==0:
                        my_lentot=int(nk*nbnd/size)
                        ave_lentot=my_lentot
                        lastrank=size-1
                else:
                        if rank==0:
                                print("Warnings:# of MPI tasks are not optimal, nbnd*nk=",nk*nbnd)
                        sys.stdout.flush()
                        my_lentot=int(nbnd*nk/size)+1
                        lastrank=int(nbnd*nk/my_lentot)
                        ave_lentot=my_lentot
                        if rank==lastrank:
                                my_lentot=nbnd*nk-ave_lentot*lastrank
                        #here my_lentot denotes total number of data one process has
                #get my_lenk,my_lenb,my_startk and my_startb
                my_starttot=ave_lentot*rank
                my_startk=int(my_starttot/nbnd)
                my_lenk=int((my_lentot+my_starttot-1)/nbnd)-my_startk+1
                my_endk=my_startk+my_lenk
                my_startb=np.zeros(my_startk+my_lenk,dtype=int)
                my_endb=np.zeros(my_startk+my_lenk,dtype=int)
                for j in range(my_startk,my_endk):
                        if j==my_startk:
                                my_startb[j]=int(my_starttot-my_startk*nbnd)
                        else:
                                my_startb[j]=int(0)
                for j in range(my_startk,my_endk):
                        if j==my_endk-1:
                                my_endb[j]=int(my_starttot+my_lentot-(my_startk+my_lenk-1)*nbnd)
                        else:
                                my_endb[j]=nbnd
                for i in range(my_startk,my_endk):
                        print('rank:',rank,'compute ik:',i,' ibnd:',my_startb[i],' to ',my_endb[i])
                if rank==0:
                        print('lastrank:',lastrank)
                sys.stdout.flush()
        if rank<=lastrank:
                for i in range(nst):
                        iist=istart+i*(nk*nbnd+1)
                        info=infil[iist].split()
                        if 'p' in info[3].lower():
                                tau=pos[int(info[1])-1]
                                for j in range(my_startk,my_endk):
                                        for l in range(my_startb[j],my_endb[j]):
                #               for j in range(1):
                #                       for l in range(1):
                                                efr=eph+kbarp[j,l,0].real-evac
                                                if efr<=0:
                                                        kbarp[j,l,1]+=0
                                                        print('found efr<0,k',j,'bnd',l)
                                                        break
                                                else:
                                                #sqrt(2me*e)/hbar*0.529177*10^(-10)=0.271106
                                                        kfr=math.sqrt(efr)*0.271106
                                                        kfz=math.sqrt(kfr**2-np.linalg.norm(kp[j])**2)
                                                        kf=[kp[j,0],kp[j,1],kfz]
                                                        sys.stdout.flush()
                                                        re=float(infil[iist+j*nbnd+l+1].split()[2])
                                                        im=float(infil[iist+j*nbnd+l+1].split()[3])
                                                        if int(infil[iist].split()[6])==1:
                                                                sys.stdout.flush()
                                                                pref=cmath.exp(-1j*np.dot(kf,tau))*(tran('pz',pol,kf)+np.dot(pol,tau)*S('pz',kf))

                                                                if (j==0)&(l==0):
                                                                        print('pz')
                                                                        print(re)
                                                                        print(im)
                                                                        print('pref:',pref)
                                                                        sys.stdout.flush()
                                                                kbarp[j,l,1]+=complex(re,im)*pref
                                                        if int(infil[iist].split()[6])==2:
                                                                pref=cmath.exp(-1j*np.dot(kf,tau))*(tran('px',pol,kf)+np.dot(pol,tau)*S('px',kf))
                                                                kbarp[j,l,1]+=complex(re,im)*pref
                                                        if int(infil[iist].split()[6])==3:
                                                                pref=cmath.exp(-1j*np.dot(kf,tau))*(tran('py',pol,kf)+np.dot(pol,tau)*S('py',kf))
                                                                kbarp[j,l,1]+=complex(re,im)*pref
                        elif 's' in info[3].lower():
                                tau=pos[int(info[1])-1]
                                for j in range(my_startk,my_endk):
                                        for l in range(my_startb[j],my_endb[j]):
                #               for j in range(5):
                #                       for l in range(5):
                                                efr=eph+kbarp[j,l,0].real-evac
                                                if efr<=0:
                                                        kbarp[j,l,1]+=0
                                                        print('found efr<0,k',j,'bnd',l)
                                                        break
                                                else:
                                                #sqrt(2me*e)/hbar*0.529177*10^(-10)=0.271106
                                                        kfr=math.sqrt(efr)*0.271106
                                                        kfz=math.sqrt(kfr**2-np.linalg.norm(kp[j])**2)
                                                        kf=[kp[j,0],kp[j,1],kfz]
                                                        sys.stdout.flush()
                                                        re=float(infil[iist+j*nbnd+l+1].split()[2])
                                                        im=float(infil[iist+j*nbnd+l+1].split()[3])
                                                        pref=cmath.exp(-1j*np.dot(kf,tau))*(tran('s',pol,kf)+np.dot(pol,tau)*S('s',kf))
                                                        sys.stdout.flush()
                                                        kbarp[j,l,1]+=complex(re,im)*pref
                                                        if (j==6)&(l==0):
                                                                print('s')
                                                                print(re)
                                                                print(im)
                                                                print('pref:',pref)
                                                                print('kbarp=',kbarp[j,l,1])
                                                                sys.stdout.flush()


        kbarp0=np.copy(kbarp)
        comm.Barrier()
        comm.Reduce(kbarp,kbarp0,MPI.SUM,0)
        if rank==0:
                for j in range(nk):
                        for l in range(nbnd):
                                kbarp0[j,l,1]=kbarp0[j,l,1]*kbarp0[j,l,1].conjugate()#*eph**2
                wfil=open('arpesbandxz','w')
                wfil.write('    ik         eb(eV)                     intensity\n')
                for i in range(nbnd):
                        for j in range(nk):
                                wfil.write('{:>6.4f}{:>15.9f}{:>30.9f}\n'.format(klist[j],kbarp[j,i,0].real,kbarp0[j,i,1].real))
                        wfil.write('\n')
                wfil.close()