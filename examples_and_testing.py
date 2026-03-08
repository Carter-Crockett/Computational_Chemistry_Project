from sympy_wavefunctions import *
from laplacian import *
from integrals_and_r12expansion import *
from Hartree_Fock import *

#This module contains examples/tests for wavefunctions, coloumbic/exchange integrals, and hartree fock


#####Defining a lot of symbolic variables here for general testing purposes#####
Z,Z1,Z2,Z_1,Z_2,Z3,Z4=sympy.symbols('Z Z1 Z2 Z_1 Z_2,Z3,Z4',real=True,positive=True,constant=True,integer=True)
a=sympy.symbols('a',real=True,positive=True,constant=True)
n=sympy.symbols('n',real=True,positive=True,constant=True,integer=True)
zeta=sympy.symbols(r'\zeta', real=True,positive=True,constant=True)
l,m=sympy.symbols('l m',real=True,nonnegative=True,integer=True)
theta,theta1,theta2,phi,phi1,phi2,r,r1,r2,rg,rl=sympy.symbols(r'\theta \theta1 \theta2 \phi \phi1 \phi2 r r1 r2 rg rl',real=True,nonnegative=True)
r3=sympy.symbols('r3',positive=True,real=True)
alpha,beta=sympy.symbols('alpha beta',real=True)
x,y,z = sympy.symbols('x y z')
hbar =sympy.Symbol(r'\bar{h}') 
mu, nu, lam, sigma, epsilon=sympy.symbols(r'\mu \nu \lambda \sigma \epsilon')
psi=sympy.Symbol(r'\psi')
c=sympy.symbols('c',real=True)
W=sympy.symbols('W')
b1,c1,d1=sympy.symbols('b1 c1 d1', real=True, positive=True)
#####




"Wavefunction examples"
#My sympy_wavefunctions module allows construction of STOs,GTOs,and hydrogen-like orbitals (hlos) as sympy objects and take args as (n,l,m,zeta,r,theta,phi)
#For example, a 1s STO orbital could be constructed as

test_STO=STO(1,0,0,zeta,r,theta,phi)

#Sympy uses method .expand() with kwarg func=True to evaluate functions
test_STO.expand(func=True)
#All of the wavefunctions have separate attributes for the spherical and radial components which can be called by .Y and .R respectively
test_STO.Y
test_STO.R
test_STO.Y.expand(func=True)
test_STO.R.expand(func=True)


"Coloumbic/Exchange integral examples"
#These functions are in my integrals_and_r12expansion module

#QMintegral and sympy_QMintegral will calculate coloumbic and exchange integrals using the physics convention of <ab||cd>=<ab|cd>-<ab|dc>. QMintegral uses scipy but requires specifying Z/zeta as a parameter.
#For example, lets calculate the values for a 1s2s coloumbic integral, 1s2s exchange integral, 1s2p coloumbic integral, and 1s2p exchange integral in that order using actual hydrogen like orbitals
#The value of these integrals can be found compared against those in literature or the ones provided here http://hitoshi.berkeley.edu/221B-S02/sol5.pdf
#hlo1 takes arguments in the order n,l,m,Z,r,theta,phi

psi1=hlo1(1,0,0,Z,r1,theta1,phi1) #1s hydrogen-like orbital for particle 1
psi2=hlo1(2,0,0,Z,r2,theta2,phi2) #2s orbital for particle 2
psi3=hlo1(2,0,0,Z,r1,theta1,phi1) #2s orbital for particle 1
psi4=hlo1(1,0,0,Z,r2,theta2,phi2) #1s orbital for particle 2

sympy_QMintegral(psi1,psi2,psi1,psi2) #This is the 1s2s coloumbic integral. It outputs a value of (17*Z)/(81) which matches the value from http://hitoshi.berkeley.edu/221B-S02/sol5.pdf
sympy_QMintegral(psi1,psi2,psi3,psi4) #This is the 1s2s exchange integral. It outputs (16*Z)/(729) which matches the value from http://hitoshi.berkeley.edu/221B-S02/sol5.pdf

psi5=hlo1(2,1,0,Z,r1,theta1,phi1) #2p orbital for particle 1
psi6=hlo1(2,1,0,Z,r2,theta2,phi2) #2p orbital for particle 2

sympy_QMintegral(psi1,psi6,psi1,psi6) #This is the 1s2p coloumbic integral. It outputs (59*Z)/(243) which is correct
sympy_QMintegral(psi1,psi6,psi5,psi4) #This is the 1s2p exchange integral. It outputs (112*Z)/(6561) which is correct


"Restricted Hartree Fock examples"
#This is stored in my Hartree_Fock module
#RHF take args in the order N,spatial_basis,Coefficient_Matrix,Z where N is the number of atomic orbitals, spatial_basis is a list of orbitals e.g. STOs, Coefficient_Matrix has dimensions of len(spatial_basis) by N and is an initial guess, Z is the number of protons

#For example, a double zeta basis for helium using STOs can be obtained from Ira Levine's 'Quantum Chemistry' 5th with zeta1=1.46 and zeta2=2.91. The final results from this match the results from his book pages 426-434
helium1=STO(1,0,0,1.46,r,theta,phi) #1s STO
helium2=STO(1,0,0,2.91,r,theta,phi) #1s STO
helium_basis=[helium1,helium2] #put basis in list
C_helium=sympy.ones(2,1) #Initial guess. There are better choices but this is fine for a simple calculation
RHF(1,helium_basis,C_helium,2) #The final calculated HF energy is -2.861662.... which is close to the accepted Hartree limit of -2.8617. The calculation takes less than half a second on my computer


###For lithium###
#Clementi and Rossi famously optimized double zeta parameters for STOs in their article "Simple basis sets for molecular wavefunctions containing atoms from Z= 2 to Z= 54." The Journal of Chemical Physics 60.12 (1974): 4725-4729.
#Unfortunately, those values now seem to be locked behind a paywall. Googling seems to suggest that the parameters they used were similar to the ones here
#This basis only uses 1s and 2s orbitals with no 2p orbitals mixed in
lithium1=STO(1,0,0,4.62,r,theta,phi) #1s STO
lithium2=STO(1,0,0,2.46,r,theta,phi) #1s STO
lithium3=STO(2,0,0,1.96,r,theta,phi) #2s STO
lithium4=STO(2,0,0,0.672,r,theta,phi) #2s STO
lithium_basis=[lithium1,lithium2,lithium3,lithium4]
C_lithium=sympy.ones(4,2)
RHF(2,lithium_basis,C_lithium,3) #The final calculated HF energy is -7.397... 
#which is okay but not great compared to the hartree limit of -7.4327. 
#This is likely a result of suboptimal parameters and no consideration of the p orbitals. The overall calculation takes around 7 seconds on my computer

###For beryllium###
#The clementi and rossi parameter values are locked behind a paywall. Googling seems to suggest that the parameters they used were similar to the ones here
#This basis only uses 1s and 2s orbitals with no 2p orbitals mixed in
beryllium1=STO(1,0,0,5.591,r,theta,phi)
beryllium2=STO(1,0,0,3.355,r,theta,phi)
beryllium3=STO(2,0,0,1.011,r,theta,phi)
beryllium4=STO(2,0,0,0.610,r,theta,phi)
beryllium_basis=[beryllium1,beryllium2,beryllium3,beryllium4]
C_beryllium=sympy.ones(4,2)
RHF(2,beryllium_basis,C_beryllium,4) #The final calculated HF energy is -14.57237.... which is in excellent agreement to the accepted Hartree limit of -14.573! The overall calculation takes about 7 seconds on my computer






###For neon###
#Same deal as the others for the parameters
#This uses 2 1s orbitals, 2 2s orbitals, and 2 2p orbitals in total
neon1=STO(1,0,0,10.6050,r,theta,phi)
neon2=STO(1,0,0,8.2869,r,theta,phi)
neon3=STO(2,0,0,3.5188,r,theta,phi)
neon4=STO(2,0,0,2.1869,r,theta,phi)
neon5=STO(2,1,0,3.1235,r,theta,phi)
neon6=STO(2,1,0,1.7656,r,theta,phi)
neon_basis=[neon1,neon2,neon3,neon4,neon5,neon6]
C_neon=sympy.ones(6,3)
RHF(3,neon_basis,C_neon,10) #This outputs a final HF energy of -124.017.... Hartrees. This is also somewhat slow taking almost 30 seconds
#This is in comparison of the HF limit of -127 to -128 (couldn't get a consistent answer). This is most likely due to suboptimal parameters and using only one atomic orbital for 2p but, overall, isn't bad for a simple RHF calc





###Open Shell###
#Currently working on adding energy ROHF calculations for open shell elements like carbon. Currently, for open shells RHF double counts some of the integrals and provides an energy below the HF limit because of it
#For example, Carbon produces the following:
carbon1=STO(1,0,0,7.68,r,theta,phi)
carbon2=STO(1,0,0,5,r,theta,phi)
carbon3=STO(2,0,0,1.98,r,theta,phi)
carbon4=STO(2,0,0,1.24,r,theta,phi)
carbon5=STO(2,1,0,2.20,r,theta,phi)
carbon6=STO(2,1,0,0.96,r,theta,phi)
carbon_basis=[carbon1,carbon2,carbon3,carbon4,carbon5,carbon6]
C_carbon=sympy.ones(6,3)
RHF(3,carbon_basis,C_carbon,6) #This outputs a final HF energy of -38.8467... which is below the HF limit of -37.69... This is because RHF is double counting some values for open shell elements



####In addition to RHF, there is RHFtest which outputs all of the intermittent data####
##The data from RHFtest for Carbon is put here for debugging purpose when I get a chance to add ROHF calculations for open shell atoms##
#Using RHFtest(3,carbon_basis,C_carbon,6) outputs the following:

#The eigenvalues for the iteration 1 is: [-14.34582544  -2.14183143  -1.77592674  -0.31725485  -0.20633362 18.69216584]
#iteration 2: [-10.06232556  -0.56041309  -0.33263225   0.3932947    0.80775805 24.32391302]
#iteration 10: [-10.85456212  -0.87629684  -0.53801691   0.32513068   0.77591218 23.63047102]


# The core energy matrix H is: Matrix([[-16.5888000000000, -5.56301154658122*sqrt(10), -2.59759611112835*sqrt(3), -1.12803295041408*sqrt(3), 0.78262667676585*pi, 0.128724861144004*pi], 
#[-5.56301154658122*sqrt(10), -35/2, -0.974607619793597*sqrt(30), -0.465713011339273*sqrt(30), 0.244802991181691*sqrt(10)*pi, 0.0449376724093362*sqrt(10)*pi], 
#[-2.59759611112834*sqrt(3), -0.974607619793597*sqrt(30), -5.28660000000000, -3.89985683678401, 0.18074636738481*sqrt(3)*pi, 0.0653398017296517*sqrt(3)*pi], 
#[-1.12803295041408*sqrt(3), -0.465713011339273*sqrt(30), -3.89985683678402, -3.46373333333333, 0.100650128230733*sqrt(3)*pi, 0.0483997366195499*sqrt(3)*pi], 
#[0, 0, 0, 0, -4.98666666666667, -2.69923233754569], [0, 0, 0, 0, -2.69923233754569, -2.57280000000000]])

#The final coefficient matrix is:
#This is C Matrix([[0.244558088330919, -2.63238423761463e-7, 0.000395459615799681], [0.770088723336114, 2.18935095267112e-6, 0.236679594847412], [-0.0103322038734926, -6.76157660272242e-6, -0.442577190131254], 
#[0.00677601580679010, 9.72860645053839e-7, -0.625801555720994], [-1.25322382511787e-7, -1.00554590592340, 1.62562679170610e-7], [7.11232718042855e-8, 0.00845485435939662, 7.51378701465265e-6]])

#The Repulsion matrix (tensor) is massive so I wont list it here but it can be seen be running RHFtest

#The final 'G' Matrix- Where the fock matrix 'F' is F=H+G is given by
#Matrix([[7.82733358913460, 6.77771947202713, 0.990725728878517, 0.415935097370623, 1.37149042418058e-6, 3.89843119803111e-8], [6.77771947202889, 6.86587419464044, 1.95221026340627, 0.889396962729228, 5.59341558955726e-7, 7.75513818411107e-8], 
#[0.990725728865333, 1.95221026340357, 3.83967931304545, 2.86341498131152, -2.21710227782440e-6, 2.12974989659349e-6], [0.415935097370806, 0.889396962728000, 2.86341498131118, 2.73277634156807, -9.11855275770371e-7, 2.66931320610133e-6], 
#[1.37149042418298e-6, 5.59341558962195e-7, -2.21710227782266e-6, -9.11855274874178e-7, 4.11042299923421, 2.12812250552502], [3.89843119839745e-8, 7.75513818526608e-8, 2.12974989523138e-6, 2.66931320698162e-6, 2.12812250555166, 2.38479491808786]])





