# This is my starter tools for using numpy's JAX package, which provides
# automatic differentiation for all kinds of functions, where all the primitives
# in each function are basic numpy things, for which JAX provides equivalents.
# JAX is designed to follow principles of functional programming, so the differentiable
# functions should not include side effects: they should not do assignments to any
# non-local variables nor to any local variables whose values can persist from one
# call to the next.  In fact, JAX arrays (created via jax.numpy.array) CANNOT be
# modified in place.

import numpy as np
import matplotlib.pyplot as plt
import grapher as GR
# import kd_array as KDA
import copy
import jax
import jax.numpy as jnp


# ****** Simple JAX test cases ******

def jaxf1(x,y):
    q = x**2 + 8
    z = q**3 + 5*x*y
    return z

def jaxf2(x,y):
    z = 1
    for i in range(int(y)):
        z *= (x+float(i))
    return z

def jaxf3(x,y):
    return x**y

df3a = jax.grad(jaxf3,argnums=0)
df3b = jax.grad(jaxf3,argnums=1)
df3c = jax.grad(jaxf3,argnums=[0,1])

def jaxf4(x,y):
    q = x**2 + 5
    r = q*y + x
    return q*r

df4 = jax.grad(jaxf4,argnums=[0,1])

def jaxf5(x,y):
    return jnp.array([x*y**3,x**3*y])

# Jax.jacrev => compute a Jacobian for reverse-mode autodiff.  We need a Jacobian, since jaxf5 outputs multiple
# values.
df5 = jax.jacrev(jaxf5,argnums=[0,1])

bad_news = 1

def jaxgum(x,y):
   global bad_news
   bad_news += 10
   return bad_news * x * y**2

# This does work, but it's bad practice, since dgum(1.0,1.0) gives a different value each time you call
# it.  Impure functions => loss of referential transparency.
dgum = jax.grad(jaxgum, argnums=[0,1])
dgum2 = jax.jit(dgum)  # A compiled version

def jaxhum(x,y,good_news):
    good_news += 10
    return good_news*x*y**2, good_news

# Two outputs, so we need a Jacobian for reverse-mode autodiff.
dhum = jax.jacrev(jaxhum,argnums=[0,1])
dhum2 = jax.jit(dhum)

# Testing out conditionals and iteration

def jumpinjax(x,n,switch,primes=[2,3,5,7,11]):
    switch = int(switch) # JAX Tracing requires real args, but range wants integers.
    if int(switch) == 0:
        for i in range(int(n)):
            x = x**2
    elif switch == 1:
        for p in primes:
            x = x*p
    else:   return - x
    return x

djuja = jax.grad(jumpinjax)

def jumpinjax2(x,n,switch,primes=[2,3,5,7,11]):
    n = int(n)  # JAX tracing requires reals, but
    switch = int(switch)
    if int(switch) == 0:    return ranger(x,n)
    elif switch == 1:   return primer(x,primes)
    else:   return - x
    return x

def ranger(y,m):
    for _ in range(int(m)):
        y = y**2
    return y

def primer(x,primes):
    for p in primes:
        x *= p
    return x

djuja2 = jax.grad(jumpinjax2)

def jumpinjax3(x,n,switch):
    if switch == 0: return ranger(x,n)
    elif switch == 1:
        return jnp.array([x**i for i in range(n)])
    else:   return -x

djuja3 = jax.jacrev(jumpinjax3)







