import sympy as sympy
#Author: Carter Crockett 2026 RHF project

#r,theta, and phi are the (relative) spherical coordinates with the physics convention that theta is the azimuthal angle and phi is the radial/polar angle
theta,phi,r=sympy.symbols(r'\theta  \phi  r ',real=True,nonnegative=True)

#x,y,z are the standard cartesian coordinates
x,y,z = sympy.symbols('x y z')

#The laplacian in spherical coordinates using sympy. f is a function of the spherical coordinates r,theta,phi
def sph_lap(f,r,theta,phi):
    dr2 = sympy.diff(f, r,2)
    dr = (sympy.sympify(2)/r)*sympy.diff(f, r)
    dtheta2 =(sympy.sympify(1)/r**2)* sympy.diff(f, theta,2)
    dtheta=(sympy.sympify(1)/r**2)*sympy.diff(f,theta)
    dphi2=(sympy.sympify(1)/((r**2)*(sympy.sin(theta)**2)))*sympy.diff(f,phi,2)
    return(dr2+dr+dtheta2+dtheta+dphi2)


#Laplacian in three dimensional cartesian coordinates    
def lap3(f,x,y,z):
    dx2 = sympy.diff(f,x,2)
    dy2 = sympy.diff(f,y,2)
    dz2 = sympy.diff(f,z,2)
    return(dx2+dy2+dz2)

#Laplacian in two dimensional cartesian coordinates
def lap2(f,x):
    dx2 = sympy.diff(f,x,2)
    dy2 = sympy.diff(f,y,2)
    return(dx2)


#Laplacian in one dimension
def lap1(f,x):
    dx2 = sympy.diff(f,x,2)
    return(dx2)

