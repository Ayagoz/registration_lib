cimport numpy as cnp
import numpy as np
from libc.math cimport ceil, floor, round
from libc.stdlib cimport malloc, free


ctypedef cnp.float64_t DOUBLE_t


def gradient(cnp.ndarray func):
    cdef int n = func.ndim
    if n == 1:
        return grad1d(<double*> func.data, func.shape[0])
    elif n == 2:
        
        return grad2d(<double*> func.data, func.shape[0], func.shape[1])
    elif n == 3:
        return grad3d(<double*> func.data, func.shape[0], func.shape[1], func.shape[2])
    else:
        raise ValueError('dimension of the input has to be 2 or 3')


cdef inline cnp.ndarray[DOUBLE_t, ndim=1] grad1d(double* func, int xlen):
    cdef cnp.ndarray[DOUBLE_t, ndim=1] grad

    grad = np.zeros(xlen)

    cdef int x, y

    for x in range(xlen):
        if x == 0:
            grad[x] = func[1] - func[0]
        elif x == xlen - 1:
            grad[x] = func[x] - func[x-1]
        else:
            grad[x] = 0.5 * (func[x+1] - func[x-1])

    return grad


cdef inline cnp.ndarray[DOUBLE_t, ndim=3] grad2d(double* func, int xlen, int ylen):
    cdef cnp.ndarray[DOUBLE_t, ndim=3] grad

    grad = np.zeros((2,xlen,ylen))

    cdef int x, y
    
    for x in range(xlen):
        for y in range(ylen):
            if x == 0:
                grad[0,x,y] = func[ylen + y] - func[y]
                #print('x==0, y==', y, 'grad[0,x,y] ==', func[ylen+y] - func[y], func[ylen+y], func[y])
            elif x == xlen - 1:
                grad[0,x,y] = func[x * ylen + y] - func[(x-1) * ylen + y]
            else:
                grad[0,x,y] = 0.5 * (func[(x+1) * ylen + y] - func[(x-1) * ylen + y])

            if y == 0:
                grad[1,x,y] = func[x * ylen + 1] - func[x * ylen]
            elif y == ylen - 1:
                grad[1,x,y] = func[x * ylen + y] - func[x * ylen + y-1]
            else:
                grad[1,x,y] = 0.5 * (func[x * ylen + y+1] - func[x * ylen + y-1])

    return grad


cdef inline cnp.ndarray[DOUBLE_t, ndim=4] grad3d(double* func, int xlen, int ylen, int zlen):
    cdef cnp.ndarray[DOUBLE_t, ndim=4] grad

    grad = np.zeros((3, xlen, ylen, zlen))

    cdef int x, y, z

    for x in range(xlen):
        for y in range(ylen):
            for z in range(zlen):
                if x == 0:
                    grad[0,x,y,z] = func[(ylen + y) * zlen + z] - func[y * zlen + z]
                elif x == xlen - 1:
                    grad[0,x,y,z] = func[(x * ylen + y) * zlen + z] - func[((x-1) * ylen + y) * zlen + z]
                else:
                    grad[0,x,y,z] = 0.5 * (func[((x+1) * ylen + y) * zlen + z] - func[((x-1) * ylen + y) * zlen + z])

                if y == 0:
                    grad[1,x,y,z] = func[(x * ylen + 1) * zlen + z] - func[x * ylen * zlen + z]
                elif y == ylen - 1:
                    grad[1,x,y,z] = func[(x * ylen + y) * zlen + z] - func[(x * ylen + y - 1) * zlen + z]
                else:
                    grad[1,x,y,z] = 0.5 * (func[(x * ylen + y+1) * zlen + z] - func[(x * ylen + y-1) * zlen + z])

                if z == 0:
                    grad[2,x,y,z] = func[(x * ylen + y) * zlen + 1] - func[(x * ylen + y) * zlen]
                elif z == zlen - 1:
                    grad[2,x,y,z] = func[(x * ylen + y) * zlen + z] - func[(x * ylen + y) * zlen + z - 1]
                else:
                    grad[2,x,y,z] = 0.5 * (func[(x * ylen + y) * zlen + z + 1] - func[(x * ylen + y) * zlen + z - 1])

    return grad
