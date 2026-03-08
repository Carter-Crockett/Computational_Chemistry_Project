import sympy as sympy
#Author: Carter Crockett 2026 RHF project




#n is the principal quantum number
n=sympy.symbols('n',real=True,positive=True,constant=True,integer=True)

#l and m are the azimuthal and magnetic quantum numbers
l,m=sympy.symbols('l m',real=True,nonnegative=True,integer=True)

#r,theta, and phi are the (relative) spherical coordinates with the physics convention that theta is the azimuthal angle and phi is the radial/polar angle
theta,theta1,theta2,phi,phi1,phi2,r,r1,r2,rg,rl=sympy.symbols(r'\theta \theta1 \theta2 \phi \phi1 \phi2 r r1 r2 rg rl',real=True,nonnegative=True)

#Z is used to designate the number of protons
Z,Z1,Z2,Z_1,Z_2,Z3,Z4=sympy.symbols('Z Z1 Z2 Z_1 Z_2,Z3,Z4',real=True,positive=True,constant=True,integer=True)

#a is (hbar**2) / (mu * (e')**2 ), where mu is the reduced mass and e' is the elementary charge atomic units (statcoloumbs)
a=sympy.symbols('a',real=True,positive=True,constant=True)




#Normalized (but not orthogonal) Slater Type orbital subclassed as sympy Function. Args are quantum numbers n,l,m; zeta for exponential param; r,theta,phi positional variables
class STO(sympy.Function):
    def __new__(self,n,l,m,zeta,r,theta,phi):
        obj=sympy.Function.__new__(self,n,l,m,zeta,r,theta,phi)
        n,l,m,zeta,r,theta,phi=obj.args
        obj.n,obj.l,obj.m,obj.zeta,obj.r,obj.theta,obj.phi=n,l,m,zeta,r,theta,phi
        Y=spherical_harmonic(l,m,theta,phi)
        obj.Y=Y
        R=sympy.sympify( (2*zeta) ) **(n+sympy.sympify(1)/2) * (sympy.sympify(1)/ sympy.sqrt(sympy.factorial(2*n))) * r**(n-1) * sympy.exp( sympy.sympify(((-1)*zeta*r)))
        obj.R=R
        return(obj)
    def __call__(self,n,l,m,zeta,r,theta,phi):
        return(STO(n,l,m,zeta,r,theta,phi))
    def _eval_expand_func(self,**hints):
        n,l,m,zeta,r,theta,phi=self.args
        return (self.R.expand(func=True)*self.Y.expand(func=True))

#Normalized (but not orthogonal) Gaussian Type orbital subclassed as sympy Function. Args are quantum numbers n,l,m; zeta for exponential param; r,theta,phi positional variables
class GTO(sympy.Function):
    def __new__(self,n,l,m,zeta,r,theta,phi):
        obj=sympy.Function.__new__(self,n,l,m,zeta,r,theta,phi)
        n,l,m,zeta,r,theta,phi=obj.args
        obj.n,obj.l,obj.m,obj.zeta,obj.r,obj.theta,obj.phi=n,l,m,zeta,r,theta,phi
        Y=spherical_harmonic(l,m,theta,phi)
        obj.Y=Y
        R=2*sympy.sqrt( sympy.sympify ( 2**(2*l+(sympy.sympify(3)/2)) * zeta**(l+(sympy.sympify(3)/2))) / ( sympy.factorial2(2*l+1) * sympy.sqrt(sympy.pi))) * r**l * sympy.exp((-1)*zeta*r**2)
        obj.R=R
        return(obj)
    def __call__(self,n,l,m,zeta,r,theta,phi):
        return(GTO(n,l,m,zeta,r,theta,phi))
    def _eval_expand_func(self,**hints):
        n,l,m,zeta,r,theta,phi=self.args
        return (self.R.expand(func=True)*self.Y.expand(func=True))



#Custom (complex-valued) spherical harmonic to ensure correct condon-shortley phase
class spherical_harmonic(sympy.Function):
    def __new__(self,l,m,theta,phi):
        obj=sympy.Function.__new__(self,l,m,theta,phi)
        l,m,theta,phi=obj.args
        obj.l,obj.m,obj.theta,obj.phi=l,m,theta,phi
        T1=radial_harmonic(m,phi)
        S1=azimuthal_harmonic(l,m,theta)
        obj.S=S1
        obj.T=T1
        return(obj)
    def __call__(self,l,m,theta,phi):
        return(spherical_harmonic(l,m,theta,phi))

    def _latex(self, printer,exp=None):
        l,m,theta,phi=self.args
        base="Y_{}^{}({},{})".format(l,m,theta,phi)
        if exp is None:
            return(base)
        return(base +"^{%s}" % exp)
    def _eval_expand_func(self, **hints):
        l,m,theta,phi=self.args
        T1=radial_harmonic(m,phi).expand(func=True)
        S1=azimuthal_harmonic(l,m,theta).expand(func=True)
        return(self.S.expand(func=True)*self.T.expand(func=True))




#Hydrogen schrodinger solution for polar angle phi. Probably should have called this polar_harmonic to avoid confusion
class radial_harmonic(sympy.Function):

    def __new__(self,m,phi):
        obj=sympy.Function.__new__(self,m,phi)
        m,phi=obj.args
        obj.m,obj.phi=m,phi
        return(obj)

    def _latex(self, printer,exp=None):
        m,phi=self.args
        base="T_{}({})".format(m,(phi))
        if exp is None:
            return(base)
        return(base+"^{%s}" % exp)
    def __call__(self,m,phi):
        return(radial_harmonic(m,phi))

    def _eval_expand_func(self, **hints):
        m,phi=self.args
        return((1/sympy.sqrt(2*sympy.pi))*sympy.exp(sympy.I*m*phi))

#Hydrogen schrodinger solution for azimuthal angle theta
class azimuthal_harmonic(sympy.Function):

    def __new__(self,l,m,theta):
        obj=sympy.Function.__new__(self,l,m,theta)
        l,m,theta=obj.args
        obj.l,obj.m,obj.theta=l,m,theta
        return(obj)

    def _latex(self, printer,exp=None):
        l,m,theta=self.args
        base="S_{}^{}({})".format(l,m,(theta))
        if sympy.exp is None:
            return(base)
        return(base+"^{%s}" % exp)
    def __call__(self,l,m,theta):
        return(azimuthal_harmonic(l,m,theta))

    def _eval_expand_func(self, **hints):
        l,m,theta=self.args
        expression = (sympy.sqrt((2*l + 1)/(sympy.sympify(2)) * sympy.sympify(sympy.factorial(l - m))/sympy.factorial(l + m))  * sympy.assoc_legendre(l, m, sympy.cos(theta)))
        return (expression.subs(sympy.sqrt(-sympy.cos(theta)**2 + 1), sympy.sin(theta))) #Because theta is between 0 and pi sin is positive so we can simplify the root without an absolute value



#Radial part of hydrogen-like orbital containing parameter 'a'
class radial_hlo(sympy.Function):
    def __new__(self,n,l,Z,r):
        obj=sympy.Function.__new__(self,n,l,Z,r)
        n,l,Z,r = obj.args
        obj.n,obj.l,obj.Z,obj.r=n,l,Z,r
        obj.Rc=sympy.sqrt(  ((sympy.sympify(2*Z)/((n*a)))**3) * (sympy.sympify(sympy.factorial(n-l-1))/(2*n*sympy.factorial(n+l))) )
        obj.Rf=sympy.exp((sympy.sympify(-Z*r)/(n*a))) *  ((sympy.sympify(2*Z*r)/(n*a)))**l * sympy.assoc_laguerre(n-l-1,2*l+1,(sympy.sympify(2*Z*r)/(n*a)))
        return(obj)
    def __call__(self,n,l,Z,r):
        return(radial_hlo(n,l,Z,r))
    def _latex(self,printer,exp=None):
        n,l,Z,r = self.args
        base="R_{}^{}({},{})".format(n,l,Z,r)
        if sympy.exp is None:
            return(base)
        return(base+"^{%s}" % exp)

    def _eval_expand_func(self,**hints):
        n,l,Z,r = self.args
        return(sympy.sqrt(  ((sympy.sympify(2*Z)/((n*a)))**3) * (sympy.sympify(sympy.factorial(n-l-1))/(2*n*sympy.factorial(n+l))) )  *sympy.exp((sympy.sympify(-Z*r)/(n*a))) *  ((sympy.sympify(2*Z*r)/(n*a)))**l * sympy.assoc_laguerre(n-l-1,2*l+1,(sympy.sympify(2*Z*r)/(n*a))))

#Radial part of hydrogen-like orbital without parameter 'a'
class radial_hlo1(sympy.Function):
    def __new__(self,n,l,Z,r):
        obj=sympy.Function.__new__(self,n,l,Z,r)
        obj.n,obj.l,obj.Z,obj.r,=n,l,Z,r
        n,l,r,Z = obj.args
        obj.Rc=sympy.sqrt(  ((sympy.sympify(2*Z)/((n)))**3) * (sympy.sympify(sympy.factorial(n-l-1))/(2*n*sympy.factorial(n+l))) )
        obj.Rf=sympy.exp((sympy.sympify(-Z*r)/(n))) *  ((sympy.sympify(2*Z*r)/(n)))**l * sympy.assoc_laguerre(n-l-1,2*l+1,(sympy.sympify(2*Z*r)/(n)))
        return(obj)
    def __call__(self,n,l,Z,r):
        return(radial_hlo(n,l,Z,r))
    def _latex(self,printer,exp=None):
        n,l,Z,r = self.args
        base="R_{}^{}({},{})".format(n,l,Z,r)
        if sympy.exp is None:
            return(base)
        return(base+"^{%s}" % exp)

    def _eval_expand_func(self,**hints):
        n,l,Z,r = self.args
        return(sympy.sqrt(  ((sympy.sympify(2*Z)/((n)))**3) * (sympy.sympify(sympy.factorial(n-l-1))/(2*n*sympy.factorial(n+l))) )  *sympy.exp((sympy.sympify(-Z*r)/(n))) *  ((sympy.sympify(2*Z*r)/(n)))**l * sympy.assoc_laguerre(n-l-1,2*l+1,(sympy.sympify(2*Z*r)/(n))))

#Hydrogen-like orbital containing parameter 'a'      
class hlo(sympy.Function):
    def __new__(self,n,l,m,Z,rtheta,phi):
        obj=sympy.Function.__new__(self,n,l,m,Z,r,theta,phi)
        obj.n,obj.l,obj.m,obj.Z,obj.r,obj.theta,obj.phi=n,l,m,Z,r,theta,phi
        n,l,m,Z,r,theta,phi=obj.args
        R=radial_hlo(n,l,Z,r)
        T1=radial_harmonic(m,phi)
        S1=azimuthal_harmonic(l,m,theta)
        Y=spherical_harmonic(l,m,theta,phi)
        obj.R,obj.Rc,obj.Rf,obj.S,obj.T,obj.Y=R,R.Rc,R.Rf,S1,T1,Y
        return(obj)
    def __call__(self,n,l,m,Z,r,theta,phi):
        return(hlo(n,l,m,Z,r,theta,phi))
    def _latex(self,print,exp=None):
        v=sympy.symbols('v')
        if self.l==0:
            v="s"
        if self.l==1:
            v="p"
        if self.l==2:
            v="d"
        if self.l==3:
            v="f"
        if self.l==4:
            v="g"
        base=r"\Psi({}{}_{}({},{}))".format(self.n,v,self.m,self.theta,self.phi)
        if sympy.exp is None:
            return(base)
        return(base+"^{%s}" % exp)
    def _eval_expand_func(self,**hints):
        n,l,m,Z,r,theta,phi=self.args
        return(self.R.expand(func=True)*self.Y.expand(func=True))

#Hydrogen-like orbital without parameter 'a'        
class hlo1(sympy.Function):
    def __new__(self,n,l,m,Z,r,theta,phi):
        obj=sympy.Function.__new__(self,n,l,m,Z,r,theta,phi)
        n,l,m,Z,r,theta,phi=obj.args
        obj.n,obj.l,obj.m,obj.Z, obj.r,obj.theta,obj.phi=n,l,m,Z,r,theta,phi
        R=radial_hlo1(n,l,Z,r)
        T1=radial_harmonic(m,phi)
        S1=azimuthal_harmonic(l,m,theta)
        Y=spherical_harmonic(l,m,theta,phi)
        obj.R,obj.Rc,obj.Rf,obj.S,obj.T,obj.Y=R,R.Rc,R.Rf,S1,T1,Y
        return(obj)
    def __call__(self,n,l,m,Z,r,theta,phi):
        return(hlo(n,l,m,Z,r,theta,phi))
    def _latex(self,print,exp=None):
        v=sympy.symbols('v')
        if self.l==0:
            v="s"
        if self.l==1:
            v="p"
        if self.l==2:
            v="d"
        if self.l==3:
            v="f"
        if self.l==4:
            v="g"
        base=r"\Psi({}{}_{}({},{}))".format(self.n,v,self.m,self.theta,self.phi)
        if exp is None:
            return(base)
        return(base+"^{%s}" % sympy.exp)
    def _eval_expand_func(self,**hints):
        n,l,m,r,Z,theta,phi=self.args
        return(self.R.expand(func=True)*self.Y.expand(func=True))

#TODO: The hydrogen-like parts in particular are messy and reflect a period when this project was simpler and not a 'Hartree-Fock' project. I should probably clean them up at some point


