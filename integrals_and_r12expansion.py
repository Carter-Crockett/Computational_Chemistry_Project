from sympy_wavefunctions import *
import scipy as scipy
import scipy as scipy
from scipy.linalg import eigh
from scipy.integrate import quad, dblquad, tplquad, nquad
import numpy as np
import time
#Author: Carter Crockett, 2026 RHF Project



#rg and rl are the greater and lesser radial terms used in the spherical harmonic addition theorem
rg,rl=sympy.symbols('rg,rl',real=True,nonnegative=True)

#r3 is a dummy variable used to evaluate some integrals
r3=sympy.symbols('r3',positive=True,real=True)





#this function is used to expand the r12 term found in coloumbic and exchange integrals using the spherical harmonic addition theorem
#n,l,m are the appropriate quantum numbers and therefore integers with l < n and |m| <= l
#l1,l2,l3,l4 and m1,m2,m3,m4 are from the wavefunctions being integrated and l12,m12 are from the expansion of r12
#The Wigner 3-j/clebsch gordan will be zero UNLESS the sum of m_i+m_j+m12 is zero AND |l_i-l_j| <= l12 <= |l_i+l_j|
#This function returns a list of the spherical harmonics that will need to be integrated
def r12sphericalexpansion(l1,m1,l2,m2,l3,m3,l4,m4):
    Y=spherical_harmonic(l,m,theta,phi)
    #For the integrals to be non-zero l12 needs the largest lower bound and the smallest upperbound
    #The convention used here is that l1,m1 and l3,m3 will be from wavefunctions for 'particle 1'
    #l2,m2 and l4,m4 will be from wavefunctions for 'particle 2'
    lstart=sympy.Max(abs(l1-l3),abs(l2-l4)) #This takes the larger of the two differences as a lower bound
    lstop=sympy.Min((l1+l3),(l2+l4)) #This takes the smaller of the two sums as an upper bound
    r12=[]
    m12=m3-m1 #m3-m1 will need to be equal to m2-m4 or the integral will be zero
    if m12!= m2-m4:
        return([0])

    elif lstart > lstop: 
        return([0])

    else:
        for l12 in range(lstart,lstop+1):
            r12.append((4*sympy.pi)/(2*l12+1)*(rl**l12)/(rg**(l12+1))*Y(l12,-m12,theta1,phi1)*Y(l12,m12,theta2,phi2)) #This is the spherical harmonic addition theorem where rl=r_< is the lesser of radial coordinates and rg=r_> is the greater
        return(r12)




#This integral numerically handles both coloumbic and exchange integrals in spherical coordinates as a single function
#As args it takes 4 wavefunctions from my sympy_wavefunctions module with appropriate n,l,m,z/zeta values specified. It uses the phys convention for 'double' inner products i.e. "<ab||cd>=<ab|cd>-<ab|dc>" where abcd are states and we take the adjoint of a and b.
#IMPORTANT:The positional arguments of the wavefunctions MUST be such that psi1 and psi3 are functions of r1,theta1,phi1 while psi2 and psi4 are functions of r2,theta2,phi2. I will probably fix this in the future. RHF automatically takes care of this
#Sympy's integration was slow so some of the integration is done using scipy
#Combining coloumbic and exchange integrals into a single function might make things less clear. I'll probably separate this back out in the future or at least give it a more descriptive name like 'Repulsion_Integral' or something

def QMintegral(psi1,psi2,psi3,psi4):
    r12=r12sphericalexpansion(psi1.l,psi1.m,psi2.l,psi2.m,psi3.l,psi3.m,psi4.l,psi4.m) #expresses r12 using spherical harmonic expansion theorem

    sph_int=[] #list that will contain values of the double integrals across theta and phi

    #This integrates the spherical harmonics of the wavefunctions and r12 using sympy. I might change this to scipy if sympy gets too slow
    for i in range(len(r12)):
        if r12[i] != 0:
            sph_int.append(sympy.integrate #phi2 integral
                           (sympy.integrate #theta2 integral
                            (sympy.integrate #phi1 integral
                             (sympy.integrate #theta1 integral
                              ( sympy.sin(theta1) * sympy.conjugate(psi1.Y).expand(func=True)* psi3.Y.expand(func=True) * r12[i].expand(func=True),(theta1,0,sympy.pi)),(phi1,0,2*sympy.pi)) #integrate spherical harmonic product of psi1*,r12,psi3 wrt theta1,phi1
                              *sympy.sin(theta2)*sympy.conjugate(psi2.Y).expand(func=True)*psi4.Y.expand(func=True),(theta2,0,sympy.pi)),(phi2,0,2*sympy.pi))) #integrate spherical harmonic product of psi2*,r12,psi4 wrt theta2,phi2

    sph_value=sum(sph_int) #condenses list into a single sympy 'symbolic expression' so it can be lambdified for scipy integration


    #This handles the integrals for the radial coordinates using scipy. It is broken into 2 parts because the function for r12 expansion changes based on which of r1,r2 is larger
    if sph_value != 0: #occasionally sph_value is 0 depending on how QMintegral is called. This handles that case


        #This is the integral as r1 goes from 0 to r2 and r2 goes from 0 to infinity. In this case, the lesser r, named 'rl', is r1 and the greater r, named 'rg', is r2
        int1=sph_value.subs(rl,r1).subs(rg,r2)* r1**2 *psi1.R.expand(func=True)*psi3.R.expand(func=True) * r2**2*psi2.R.expand(func=True)*psi4.R.expand(func=True) #subs values for rg/rl
        int1lam=sympy.lambdify([r1,r2],int1) #converts sympy expression to something compatible with scipy
        integral1=dblquad (int1lam, 0, +np.inf, lambda r2:0, lambda r2 : r2 ) #I know the lambda looks backwards but it's correct for scipy integration.


        #This is the integral as r1 goes from r2 to infinity and r2 goes from 0 to infinity. In this case, the lesser r, named 'rl', is r2 and the greater r, named 'rg', is r1
        int2=sph_value.subs(rl,r2).subs(rg,r1)* r1**2* psi1.R.expand(func=True)*psi3.R.expand(func=True) * r2**2*psi2.R.expand(func=True)*psi4.R.expand(func=True)
        int2lam=sympy.lambdify([r1,r2],int2)
        integral2=dblquad (int2lam, 0, +np.inf, lambda r2:r2, lambda r2 : +np.inf )

        final=integral1[0]+integral2[0]

        return(final)
    else:
        return(0) #returns 0 if all the angular integrals had a value of 0


#Same as QMintegral but with integration times included for testing optimizations
def QMintegraltest(psi1,psi2,psi3,psi4):
    r12=r12sphericalexpansion(psi1.l,psi1.m,psi2.l,psi2.m,psi3.l,psi3.m,psi4.l,psi4.m) 

    sph_int=[] 

    start=time.time()
    for i in range(len(r12)):
        sph_int.append(sympy.integrate 
                       (sympy.integrate 
                        (sympy.integrate 
                         (sympy.integrate 
                          ( sympy.sin(theta1) * sympy.conjugate(psi1.Y).expand(func=True)* psi3.Y.expand(func=True) * r12[i].expand(func=True),(theta1,0,sympy.pi)),(phi1,0,2*sympy.pi)) 
                          *sympy.sin(theta2)*sympy.conjugate(psi2.Y).expand(func=True)*psi4.Y.expand(func=True),(theta2,0,sympy.pi)),(phi2,0,2*sympy.pi))) 
    stop=time.time
    print("The time to do the spherical harmonic integrals was {} seconds".format(stop-start))
    sph_value=sum(sph_int) 



    if sph_value != 0: 


        start=time.time()
        int1=sph_value.subs(rl,r1).subs(rg,r2)* r1**2 *psi1.R.expand(func=True)*psi3.R.expand(func=True) * r2**2*psi2.R.expand(func=True)*psi4.R.expand(func=True) 
        int1lam=sympy.lambdify([r1,r2],int1) 
        integral1=dblquad (int1lam, 0, +np.inf, lambda r2:0, lambda r2 : r2 )
        stop=time.time()
        print("The time for radial integral 1 was {}".format(stop-start))


        start=time.time()
        int2=sph_value.subs(rl,r2).subs(rg,r1)* r1**2* psi1.R.expand(func=True)*psi3.R.expand(func=True) * r2**2*psi2.R.expand(func=True)*psi4.R.expand(func=True)
        int2lam=sympy.lambdify([r1,r2],int2)
        integral2=dblquad (int2lam, 0, +np.inf, lambda r2:r2, lambda r2 : +np.inf )
        stop=time.time()
        print("The time for radial integral 2 was {}".format(stop-start))
        final=integral1[0]+integral2[0]

        return(final)
    else:
        return(0) 





###older versions that purely uses sympy###
#left these two in for examples

#this r12 expansion only works for single inner products and is fine for simple coloumbic and exchange integrals
#Had to make the other r12 expansion to accommodate the more complicated integrals that show up in Hartree Fock
def sympy_r12sphericalexpansion(l1,m1,l2,m2):
    Y=spherical_harmonic(l,m,theta,phi)
    lstop=l1+l2
    m3=-(m1+m2)
    r12=[]
    for l3 in range(0,lstop+1):
        r12.append((4*sympy.pi)/(2*l3+1)*(rl**l3)/(rg**(l3+1))*Y(l3,-m3,theta1,phi1)*Y(l3,m3,theta2,phi2))
    return(r12)

#The other QMintegral uses scipy for the radial integrations
#I had to use a dummy variable 'r3' to get it to analytically evaluate these integrals. But, it's fairly fast for simple coloumbic/exchange integrals which was the original goal of this project.
def sympy_QMintegral(psi1,psi2,psi3,psi4):
    r12=sympy_r12sphericalexpansion(psi1.l,psi1.m,psi2.l,psi2.m)
    sph_int=[]
    rad_int=[]
    r3=sympy.symbols('r3',positive=True)
    for i in range(len(r12)):
        sph_int.append(sympy.integrate(sympy.integrate(sympy.integrate(sympy.integrate( sympy.sin(theta1) * sympy.conjugate(psi1.Y).expand(func=True)* psi3.Y.expand(func=True) * r12[i].expand(func=True),(theta1,0,sympy.pi)),(phi1,0,2*sympy.pi))*sympy.sin(theta2)*sympy.conjugate(psi2.Y).expand(func=True)*psi4.Y.expand(func=True),(theta2,0,sympy.pi)),(phi2,0,2*sympy.pi)))
    list2=sum(sph_int)
    integral1=sympy.integrate(list2.subs(rl,r1).subs(rg,r2)* r1**2 *psi1.R.expand(func=True)*psi3.R.expand(func=True),(r1,0,r3))
    integral2=sympy.integrate(list2.subs(rl,r2).subs(rg,r1)* r1**2* psi1.R.expand(func=True)*psi3.R.expand(func=True),(r1,r3,sympy.oo))
    integral3=(integral1+integral2).subs(r3,r2)
    integral4=sympy.integrate(integral3* r2**2 *psi2.R.expand(func=True)*psi4.R.expand(func=True),(r2,0,sympy.oo))
    return(integral4)



