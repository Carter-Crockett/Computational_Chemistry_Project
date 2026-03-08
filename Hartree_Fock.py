from sympy_wavefunctions import *
from laplacian import *
from integrals_and_r12expansion import *
import sympy as sympy
import numpy as np
import scipy as scipy
from scipy.linalg import eigh
from scipy.integrate import quad, dblquad, tplquad, nquad
#Author:Carter Crockett, 2026 RHF project





#Indexes used for HF. lambda is abbreviated as 'lam' to avoid confusion with python lambda functions
#The convention used here is that mu,nu,lam,sigma refer to specific basis functions. Epsilon is the orbital energy (i.e. the eigenvalue)
mu, nu, lam, sigma, epsilon=sympy.symbols(r'\mu \nu \lambda \sigma \epsilon')


#Calculates the Core Energy of a Rootheran-Hall Basis for atomic wavefunctions
#As args spatial_basis is a list of wavefunctions constructing using my sympy_wavefunctions module with appropriate n,l,m,Z/zeta values specified and positional args left as r,theta,phi. Z is the number of protons
#Uses sympy to do the integration. Might change to scipy to increase speed for large basis
#TODO: many of these integrals are equivalent. Should add symmetry rules
def Core_Energy(spatial_basis,Z):
    spb=spatial_basis #shorthand name to save time
    n=len(spatial_basis)
    H=sympy.zeros(n,n) #intialize matrix
    for mu in range(n):
        for nu in range(n):
            Psi1=spb[mu].expand(func=True)
            Psi2=spb[nu].expand(func=True)
            Hkinetic=sympy.integrate(  sympy.integrate(   sympy.integrate(   ( sympy.sympify(-1)/(2) * sympy.conjugate(Psi1) * sph_lap( Psi2,r,theta,phi) * sympy.sin(theta))  , (theta,0,sympy.pi)),   (phi,0,2*sympy.pi)) * r**2 , (r,0,sympy.oo)) #The kinetic energy
            Hattract=sympy.integrate(  sympy.integrate(   sympy.integrate(   ( (-1)*Z* sympy.conjugate(Psi1) * Psi2 * sympy.sin(theta))  , (theta,0,sympy.pi)),   (phi,0,2*sympy.pi)) * r , (r,0,sympy.oo)) #The attractive energy between electrons and nucleus
            H[mu,nu]=Hkinetic+Hattract
    return(H)

#Calculates the overlap to account for a non-orthonormal basis
#As args spatial_basis is a list of wavefunctions constructing using my sympy_wavefunctions module with appropriate n,l,m,Z/zeta values specified and positional args left as r,theta,phi.
#Uses sympy to do the integration
#TODO: many of these integrals are equivalent. I might add symmetry rules but the integrals are fairly quick
def Overlap_Matrix(spatial_basis):
    spb=spatial_basis #shorthand name
    n=len(spatial_basis)
    S=sympy.zeros(n,n) #initialize matrix
    for mu in range(n):
        for nu in range(n):
            Psi1=spb[mu].expand(func=True)
            Psi2=spb[nu].expand(func=True)
            Sint=sympy.integrate(  sympy.integrate(   sympy.integrate(   ( sympy.conjugate(Psi1) * Psi2*sympy.sin(theta))  , (theta,0,sympy.pi)),   (phi,0,2*sympy.pi)) * r**2 , (r,0,sympy.oo))
            S[mu,nu]=Sint
    return(S)

#Density Matrix for Hartree Fock
#The Coefficient_Matrix has elements C_(mu)(i) where mu corresponds to a spatial basis function and 'i' corresponds to an atomic orbital i.e. the product of a spatial and spin orbital
#The coefficient matrix has a row for every spatial basis function and a column for every atomic orbital
def Density_Matrix(Coefficient_Matrix):
    C1=Coefficient_Matrix
    M1=C1
    M2=C1.adjoint()
    P=M1*M2
    return(P)

#Matrix (tensor) for the coloumbic and exchange integrals
#As args spatial_basis is a list of wavefunctions constructing using my sympy_wavefunctions module with appropriate n,l,m,Z/zeta values specified and positional args left as r,theta,phi.
#TODO: many of these integrals are equivalent. For a closed shell we only need to sum over n/2 rather than n and then multiply by two. I left this is as for now to use for open shell calculations. 
#DZ basis for 2nd row elements with p-orbitals included can take almost thirty seconds will optimize at some point
def Repulsion_Matrix(spatial_basis):
    spb=spatial_basis #shorthand name
    n=len(spatial_basis)
    R=sympy.MutableDenseNDimArray.zeros(n,n,n,n) #initialize tensor for storing the values of the coloumbic and exchange integrals
    for mu in range(n):
        for nu in range(n):
            for lam in range(n): 
                for sigma in range(n):
                    #The coloumbic and exchange integrals are handled by QMintegral from my integrals_and_r12expansion module
                    #QMintegral uses the physics convention for <ab||cd>. This next part takes care of changing the positional arguments to reflect that
                    Psi1=spb[mu].subs(r,r1).subs(theta,theta1).subs(phi,phi1)
                    Psi2=spb[lam].subs(r,r2).subs(theta,theta2).subs(phi,phi2)
                    Psi3=spb[nu].subs(r,r1).subs(theta,theta1).subs(phi,phi1)
                    Psi4=spb[sigma].subs(r,r2).subs(theta,theta2).subs(phi,phi2)

                    Psi5=spb[nu].subs(r,r2).subs(theta,theta2).subs(phi,phi2)
                    Psi6=spb[sigma].subs(r,r1).subs(theta,theta1).subs(phi,phi1)

                    R[mu,nu,lam,sigma]=R[mu,nu,lam,sigma]+(   ( 2*QMintegral( Psi1,Psi2,Psi3,Psi4) - QMintegral( Psi1,Psi2,Psi6,Psi5) )  ) #The first integral is a coloumbic integral the second is an exchange integral


    return(R)

#This matrix multiplies the repulsion tensor by the appropriate density matrix elements
#This is called to 'update' the contribution of the coefficient matrix in RHF
def G_Matrix(spatial_basis,Repulsion_Matrix,Coefficient_Matrix):
    spb=spatial_basis
    n=len(spatial_basis)
    R=Repulsion_Matrix
    P=Density_Matrix(Coefficient_Matrix)
    G=sympy.zeros(n,n)
    for mu in range(n):
        for nu in range(n):
            for lam in range(n):
                for sigma in range(n):
                     G[mu,nu]=G[mu,nu]+(  P[lam,sigma]*R[mu,nu,lam,sigma])
    return(G)


#Restricted Hartree Fock. This is the function to call to do RHF calculations
##As arguments, it needs the following:
##'N' the number of atomic orbitals
## spatial_basis is a list of wavefunctions constructing using my sympy_wavefunctions module with appropriate n,l,m,Z/zeta values specified and positional args left as r,theta,phi.
##Coefficient_Matrix is an len(spatial_basis) by N matrix. An initial guess needs to be made. For row 1 and 2 elements sympy.ones(len(spatial_basis),N) is good enough
##Z is the number of protons in the element
#TODO: This can be slow because of the repulsion_matrix

def RHF(N,spatial_basis,Coefficient_Matrix,Z):
    start=time.time()
    spb=spatial_basis
    n=len(spb)
    C=Coefficient_Matrix
    H=Core_Energy(spatial_basis,Z)
    R=Repulsion_Matrix(spb)
    S=Overlap_Matrix(spatial_basis)
    G=G_Matrix(spb,R,C)
    F=H+G #Fock matrix
    F1=(sympy.matrix2numpy(F)).astype(float) #Fock and overlap matrices need to be converted to numeric numpy arrays to be compatible with scipys eigh function. matrix2numpy converts them to a numpy object array and .astype float changes the type to numeric
    S1=(sympy.matrix2numpy(S)).astype(float)

    #This solves the Rootheran-Hall general eigenvalue problem FC=SCE. eigh returns two arrays with the eigenvalues ordered by size (smallest first)
    #It requires S to be positive definite so issues may occur if the basis is large enough for linear dependence.
    eigenvalues,eigenvectors=eigh(F1,S1) 


    for i in range(10): #right now it just iterates 10 times. Will eventually change it to iterate until eigenvalue convergence
        Clist=[]
        print("this is iteration",i+1)
        for j in range(N):
            Clist.append(sympy.sympify(eigenvectors).tomatrix().col(j)) #This grabs the first N eigenvectors where N is the number of atomic orbitals

        C=sympy.Matrix.hstack(*Clist) #This creates a new sympy coefficient matrix using the calculated eigenvectors

        G=G_Matrix(spb,R,C)
        F=H+G

        F1=sympy.matrix2numpy(F).astype(float)
        eigenvalues,eigenvectors=eigh(F1,S1)
        print("The eigenvalues are",eigenvalues)



    P=Density_Matrix(C) #Density matrix using the final converged Coefficient Matrix
    TE=P*(H+F) #Total energy is the trace of P*(H+F)
    print("The HF Energy is {}".format(float(TE.trace())))
    stop=time.time()
    print("The overall process took",stop-start)

#The same as RHF but it shows each individual matrix for debugging
def RHFtest(N,spatial_basis,Coefficient_Matrix,Z):
    start=time.time()
    spb=spatial_basis
    n=len(spb)
    C=Coefficient_Matrix

    H=Core_Energy(spatial_basis,Z)
    print("This is H {}".format(H))

    R=Repulsion_Matrix(spb)
    print("This is R {}".format(R))

    S=Overlap_Matrix(spatial_basis)
    print("This is S {}".format(S))

    G=G_Matrix(spb,R,C)
    print("This is G {}".format(G))

    F=H+G
    print("This is F {}".format(F))

    F1=(sympy.matrix2numpy(F)).astype(float) 
    S1=(sympy.matrix2numpy(S)).astype(float)

    eigenvalues,eigenvectors=eigh(F1,S1) 
    for i in range(10): 
        Clist=[]
        print("this is iteration",i+1)
        for j in range(N):
            Clist.append(sympy.sympify(eigenvectors).tomatrix().col(j)) 
        C=sympy.Matrix.hstack(*Clist) 
        print("This is C {}".format(C))

        G=G_Matrix(spb,R,C)
        print("This is G {}".format(G))
        F=H+G

        F1=sympy.matrix2numpy(F).astype(float)
        eigenvalues,eigenvectors=eigh(F1,S1)
        print("The eigenvalues are",eigenvalues)



    P=Density_Matrix(C) 
    TE=P*(H+F) 
    print("The HF Energy is {}".format(float(TE.trace())))
    stop=time.time()
    print("The overall process took",stop-start)
    return(H,C,R,S,G,F,P)

#TODO: RHF works for closed shell calculations. Will add ROHF for open shells optimize RHF to sum over n/2 instead of n


