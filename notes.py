import numpy 

#print(numpy.__version__)

#------------------------------------

A=numpy.full(10,numpy.nan) #array of nans
B=numpy.empty(10) #array of 10 random elements
C=numpy.random.randint(0,10,10) #array of random 10 integers in the given range,excluding the right end
CC=numpy.random.randint(0,10,[3,4]) #creating a 3x4 array with elements in the given range
D=numpy.random.randn(10) #array of 10 random integers
E=numpy.random.uniform(0,10,10) #array of random 10 numbers in the given range
EE=numpy.random.uniform(0,10,[3,4]) #3x4 array of random numbers in the given range
F=numpy.random.random(10) #array of 10 random numbers
H=numpy.linspace(0,10,11,endpoint=True) # dividing the range into equal intervals
P=numpy.ones(shape=[2,3],dtype="float64") #2x3 array full of 1's 
Q=numpy.arange(10,50).reshape([4,10])

#-------------------------------------

A=numpy.array([[1,2,3],[4,5,6]],dtype="int32")
item_size=A.itemsize #number of bytes used for each element
elements=A.size #total number of elements
total_size=item_size*elements 
#print(total_size*elements)

#-------------------------------------

A=numpy.array([[5,1,0],[2,5,1],[0,1,0]])
index_of_nonzero=[list(map(int,(i,j))) for i,j in zip(*numpy.nonzero(A))] #finding the indexes of nonzero elements
#print(index_of_nonzero)                                                    by using numpy.nonzero
index_of_zero=[list(map(int,(i,j))) for i,j in zip(*numpy.nonzero(A==0))] #finding the indexes of zero elements by using a trick
#print(index_of_zero)
B=numpy.random.randint(0,10,20).reshape([5,4]).astype(float)
numpy.put(B,numpy.random.choice(range(0,20),4),[numpy.nan]) #creating an array and placing numpy.nan to 4 random positions
index_of_nan=[list(map(int,(i,j))) for i,j in zip(*numpy.nonzero(numpy.isnan(B)))] #finding the index of nan values from the array
                                                                                    #we just created
#------------------------------------

A=numpy.identity(3) #creating 3x3 identity matrix
B=numpy.eye(3,5) #creating a 3x5 matrix with 1's on diagonal
C=numpy.diag([1,2,3,4],3) #creating a matrix with 1,2,3,4 at diagonal which starts at the third column 

#------------------------------------

A=numpy.array([[1,-3,5],[3,-1,3]])
mx,mn=numpy.max(A),numpy.min(A)
mxidx=numpy.unravel_index(numpy.argmax(A),A.shape) #finding the index of max element 
mnidx=numpy.unravel_index(numpy.argmin(A),A.shape) #finding the index of min element

#------------------------------------

A=numpy.array([[2,6,1],[0,4,-1],[2,5,-2]])
mn=A.mean(axis=1,keepdims=True) #finding mean of each row
mn2=A.mean(axis=0,keepdims=False) #finding mean of each column 
mn3=numpy.atleast_2d(mn2) #we can convert the resulting 1d array to 2d by using atleast_2d method

#------------------------------------

A=numpy.random.randint(1,10,[3,3])
paddedA=numpy.pad(A,(1,2),mode="constant",constant_values=0) # padding the 2d array with the given value
B=numpy.random.randint(1,10,5)
paddedB=numpy.pad(B,(1,2),mode="constant",constant_values=(11,12)) # padding 1d array

#------------------------------------

A=numpy.random.randint(1,10,[5,6])
A[[0,-1],:]=A[:,[0,-1]]=0 # another way to create a padded array by using fancy indexing
#print(A)

#------------------------------------

A=numpy.random.randint(1,10,[3,5]).astype("float64")
numpy.put(A,numpy.random.choice(range(15),4),[numpy.nan])
print(numpy.count_nonzero(numpy.isnan(A))) # we can use count_nonzero to count False,or 0's.We can use this trick to
                                            #count the number of nan's:4

#------------------------------------

A=numpy.diag([1,2,3,4],-1) #k can also be negative.In this case,it is as if elements start from -1th row

#------------------------------------

A=numpy.array([[1,2],[3,4]]) 
#print(numpy.tile(A,2)) #replaces each element with concatenation of itself k times
B=numpy.array([[[1,2],[6,3]],[[3,0],[9,5]]])
#print(numpy.tile(B,2)) -> [[[1,2,1,2],[6,3,6,3]],[[3,0,3,0],[9,5,9,5]]]
checkerboard=numpy.tile([[0,1],[1,0]],(4,4))
#print(checkerboard)
AA=numpy.ones((8,8),dtype="int64")
AA[0::2,1::2]=AA[1::2,0::2]=0 #using fancy indexing to create checkerboard pattern
#print(AA)

#------------------------------------

A=numpy.empty((6,7,8))
idx=numpy.unravel_index(100,A.shape) #using unravel index to get the index for 100th element of a 6x7x8 array
#print(idx)

#------------------------------------

DTYPE=numpy.dtype([("color",[("r",numpy.ubyte),("g",numpy.ubyte),("b",numpy.ubyte),("a",numpy.ubyte)]),
                   ("coordinates",[("x",float),("y",float)])])
A=numpy.empty((3,4),dtype=DTYPE)
A["color"]["r"]=numpy.random.random(A.shape)
A["color"]["g"]=numpy.random.random(A.shape)
A["color"]["b"]=numpy.random.random(A.shape)
A["color"]["a"]=numpy.random.random(A.shape)
A["coordinates"]["x"]=numpy.random.random(A.shape)
A["coordinates"]["y"]=numpy.random.random(A.shape)
#print(A[0][1]["color"]["g"]) 
#This way,we can create a random dtype and access the elements by field names

#-----------------------------------

A=numpy.array([[1,5],[2,3],[4,1]])
B=numpy.array([[8,3,0],[4,1,6]])
#print(numpy.dot(A,B)) #calculating dot product of matrices

#-----------------------------------

A=numpy.random.randint(0,10,(3,4))
#print(A)
A[(A>=3) & (A<=8)]*=-1  #by using this,we can change the elements between the given interval in-place
#print(A)
A[numpy.where((A>=3) & (A<=8),True,False)]*=-1 # and this is an alternative method
#print(A)

#----------------------------------

A=numpy.array([numpy.nan]).astype(int) #casting nan as int returns a random value
B=numpy.array([numpy.nan]).astype(int) #returns nan,this is similar to why we cannot use nan in integer arrrays but we can in float arrays
C=numpy.array(0)
D=numpy.array(0)
#print(C/D) #prints nan,as expected
#print(C//D) #prints 0

#----------------------------------

A=numpy.array([1,3,5,7,9])
B=numpy.array([1,2,3,4,5])
#print(numpy.intersect1d(A,B)) #we can use intersect1d to find the intersection of two arrays
C=numpy.random.randint(3,6,(4,5))
D=numpy.random.randint(3,6,(5,7))
#print(numpy.intersect1d(C.flatten(),D.flatten())) finding the intersection of multidimensional arrays
#print(numpy.intersect1d(numpy.ravel(C),numpy.ravel(D))) another method for the same
#print(numpy.intersect1d(numpy.ravel(C),numpy.ravel(D)))

import functools
arrays=[A,B,C.flatten(),D.flatten()]
#print(functools.reduce(numpy.intersect1d,arrays)) #we can use reduce to intersect multiple arrays

#---------------------------------

A=numpy.array([[1,5,2],[4,1,0],[7,0,3],[3,0,4],[1,0,3]])
B=numpy.array([[1,3],[2,7],[7,2]])
#print(numpy.dot(A,B)) #dot function can be used to perform matrix multiplication

#---------------------------------

A=numpy.array([[1,7,4],[3,6,2],[1,5,4]])
A[numpy.where((A>=3) & (A<=7))]*=-1 
A[(A>=3) & (A<=7)]*=-1  #two different ways of performing the desired operation
#print(A)

#---------------------------------

A=numpy.datetime64("today") #today
B=numpy.datetime64("2013-07-03") #custom date
C=numpy.timedelta64(1)
#print(A-C) #going back to 1 days ago
#print(A-numpy.timedelta64(30000)) #going back to Hitler's era
D=numpy.arange(numpy.datetime64("2016-07-01"),numpy.datetime64("2016-08-01")) #finding all the dates in July
#print(D)

#--------------------------------

A=numpy.random.randint(1,10,(3,4)).astype(float)
B=numpy.random.randint(1,10,(3,4)).astype(float)
numpy.add(B,A,out=B)
numpy.divide(A,-2,out=A)
numpy.multiply(A,B,out=B)
numpy.negative(A,out=A) #changes the sign of every number in-place
C=numpy.array([[True,True],[False,True],[False,False]])
numpy.logical_not(C,out=C) #negates every value in C in-place
#print(C)
#print(B) #we can use add,multiply,divide functions and out parameter to perform in-place arithmetic operations 

#--------------------------------

A=numpy.array([[1.7,3.2],[-3.2,-4.8]],dtype="float64")
B=numpy.trunc(A) #removes floating part
C=numpy.floor(A) #gives different result,rounds off to neares smaller number
D=A-(A%1) #gives the same result as floor 
E=A//1 #gives the same result as floor
F=A.astype(int) #same result as trunc

#--------------------------------

def f():
    while True:
        yield numpy.random.randint(1,10)

A=numpy.fromiter(f(),dtype="int32",count=3)
#print(A) #we can use generators and fromiter to create numpy arrays

#--------------------------------

A=numpy.linspace(0,10,10,endpoint=False)[1:]
#print(A) #we can use endpoint parameter and exclude the first element to divide the given interval to equal parts

#--------------------------------

A=numpy.random.random(10)
#print(numpy.sort(A)) #this is not an in-place operation
A.sort() #in-place sorting
#print(A)

#--------------------------------

import functools,time
A=numpy.arange(100000)
x=time.time()
tot=numpy.sum(A)
y=time.time()
#print(y-x)
tot=A.sum()
z=time.time()
#print(z-y)
tot=functools.reduce(numpy.add,A)
k=time.time()
#print(k-z) #in general,reduce is preferable for small arrays in terms of efficiency

#---------------------------------

A=numpy.array([[1,2],[3,4],[5,6]])
B=numpy.array([[1,2,3],[4,5,6]])
#print(numpy.array_equal(A,B)) #array equality can be tested by array_equal method
#print(numpy.array_equal(A.flatten(),B.flatten())) #prints true
#print(numpy.allclose(A.flatten(),B.flatten())) #all_close can be used to ignore small differences stemming from 
                                                #the minor errors in handling floating numbers

#---------------------------------

A=numpy.random.randint(0,10,(5,3))
A.flags.writeable=False #making the numpy array read-only
#A[(1,1)]=1 would give error

#---------------------------------

A=numpy.array([[1,5,2],[5,1,0],[2,5,1]])
idx=numpy.unravel_index(numpy.argmax(A),A.shape)
A[idx]=0 #this can be used if there is only one max element

mx=A.max()
A[A==mx]=0 
#print(A) #this can be used to set all max elements to 0,if there are many max elements

#---------------------------------

A=numpy.array([1,2,3])
B=numpy.array([4,5,6])
X,Y=numpy.meshgrid(A,B) #this can be used to create 2d grid
grid=[list(map(int,(i,j))) for A,B in zip(X,Y) for i,j in zip(A,B)] #creating a grid with the every point covering the grid
#print(grid)

#---------------------------------

a=numpy.iinfo(numpy.int64).max
b=numpy.finfo(numpy.float64).max
c=numpy.iinfo(numpy.int32).min
#print(a,b,c) #numpy.iinfo and numpy.finfo can be used to get info about data types 

#---------------------------------

A=numpy.array([[1,4,2],[3,6,3],[5,1,4]])
x=3.4
dif=abs(A-x).min()
#print(A[numpy.isclose(abs(A-x),dif)]) #numpy.isclose can be used to avoid small errors enountered in handling floating point numbers
#print(A.flatten()[numpy.abs(A-x).argmin()]) # another method is using argmin

#---------------------------------

A=numpy.array([[1,5,4],[3,0,1]])
X,Y=numpy.atleast_2d(A[0,:],A[1,:])
Z=numpy.sqrt((X-X.T)**2+(Y-Y.T)**2)
#print(Z) #using this way,we can the distance between each pair of points

#--------------------------------


for idx,val in numpy.ndenumerate(A):
    #print(idx,val==A[idx]) #ndenumerate can be used to enumerate numpy array
    pass
for idx in numpy.ndindex(A.shape):
    #print(idx) #shape can be used to get indexes for the grid positions.both serve the same purpose 
    pass

#----------------------------------

A=numpy.ones((3,4))
numpy.put(A,numpy.random.choice(range(10),5),0)
#print(A) we have to use the indexes for flattened version of the array

#-----------------------------------

A=numpy.array([[1,numpy.nan,2],[3,numpy.nan,0]],dtype=float)
#print(numpy.isnan(A).all(axis=0)) #we can use this code to check whether there is a null column
B=numpy.array([[1,numpy.nan,2],[0,5,1],[1,5,numpy.nan]])
#print(~numpy.isnan(B).any(axis=0)) # we can use this code to find columns without any nan value

#-----------------------------------

A=numpy.array([[4,1,6],[1,0,4],[3,1,2]])
B=numpy.argsort(A[:,2])
#print(A[B,:]) # sorting a numpy matrix by 2th column
#print(numpy.reshape(numpy.sort(A.flatten()),A.shape)) #sorting all elements 
#print(numpy.sort(A,axis=1)) #sorting each row in itself
C=numpy.argsort(A,axis=1)
""" for i in range(A.shape[0]):
    A[i]=A[i,C[i]] #sorting each row in itself
print(A) """

#print(A[[[0],[1],[2]],[[2,1,0],[1,0,2],[0,1,2]]]) #advanced slicing 
#print(A[numpy.arange(A.shape[0])[:,None],numpy.argsort(A,axis=1)]) #using advanced slicing to sort each row in itself

#----------------------------------

A=numpy.array([[2,0,1],[4,1,2],[3,4,5]])
B=numpy.array([[0,4,5],[6,1,2],[3,4,1]])
#for i,j in numpy.nditer([A,B]):
    #print(i,j) # nditer can be used to iterate over two matrices at the same time.
    # But sizes must be able to broadcast together
C=numpy.array([[2,4],[5,1]])
iter=numpy.nditer([A,B,None])
for i,j,C in iter:
    C[...]=i+j
#print(iter.operands[2]) #returns C

#----------------------------------

A=numpy.zeros(10,dtype="int64")
Y=numpy.random.randint(0,len(A),20)
A+=numpy.bincount(Y,minlength=len(A))
#In this example,we add 1 to the elements of A whose indices are specified by Y

#----------------------------------

X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = numpy.bincount(I,weights=X) #we can use bincount with weights as well.Normally,bincount takes the weight of each element as 1
                                #and counts elements but X,here, determines the weight of each index in I
#print(F)

#----------------------------------
import itertools 
A=numpy.array([1,5,2,0,5,2,3])
#print(numpy.unique(A)) #finding the unique elements in the given numpy array
#print(numpy.unique(A).size)
B=numpy.array([[1,2,3,1],[1,2,3,1],[3,2,6,3]])
#print(numpy.unique(B,axis=0)) #remove duplicate rows
#print(numpy.unique(B,axis=1)) #removes duplicate columns
#print(list(map(numpy.unique,B))) #remove duplicates in each row
#print(list(map(numpy.unique,B.T))) #remove duplicates in each column

#----------------------------------

w,h=256,256
A=numpy.random.randint(0,4,(w,h,3)).astype(numpy.ubyte)
#print(len(numpy.unique(A.reshape(-1,3),axis=0))) #finding the number of unique colors,
                                                #which are represented as 3 8-bit integers
A24=numpy.dot(A.astype("int32"),[1,1<<8,1<<16]) #representing each color as 24-bit number,not 3 8-bit numbers
#print(A24)
#print(len(numpy.unique(A24))) #finding the number of colors

#----------------------------------

A=numpy.array([1,2,3,4,5,6])
I=numpy.array([1,0,1,0,1])
D=A[I]
#print(numpy.bincount(Y,weights=D)) #we can use this trick to calculate sum of subarrays for given index map I

#----------------------------------

A=numpy.array([1,2,3,4,5])
z=4 
B=numpy.zeros(z*(len(A)-1)+len(A))
B[::z+1]=A
#print(B) #we can use this code to insert 0's inbetween each element in A

#----------------------------------

A=numpy.array([[3,0,2],[5,1,2],[2,7,3]])
#A[[0,1]]=A[[1,0]] #swapping two rows in a numpy matrix
#print(A)
#A[:,[0,2]]=A[:,[2,0]] #swapping two columns
#print(A)

#----------------------------------

A=numpy.array([[1,1,1],[2,2,2],[3,3,3]])
B=numpy.array([[1,2,3],[1,2,3],[1,2,3]])
#print(numpy.roll(A,shift=2,axis=0)) #this can be used to shift elements in a numpy array row-wise
#print(numpy.roll(B,shift=1,axis=1)) #this can be used to shift elements in a numpy array column-wise

#----------------------------------

A=numpy.array([1,2,5,3,1,5,2])
#print(numpy.repeat(numpy.arange(len(A)),A)) #we can use repeat with frequency array.Using this trick,we can
                                    #find the array B where numpy.bincount(B)==A

#----------------------------------

A=numpy.arange(1,10)
T=numpy.lib.stride_tricks.sliding_window_view(A,4)
#print(T)
B=numpy.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
K=numpy.lib.stride_tricks.sliding_window_view(B,window_shape=3,axis=1) #we can use numpy.lib.stride_tricks.sliding_window_view
#print(K)                                                           #to get a sliding window view of element
C=numpy.arange(1,17,dtype=numpy.uint32)
#print(numpy.lib.stride_tricks.sliding_window_view(B,(2,3))) #returns every 2x3 consecutive window

#----------------------------------

A=numpy.arange(1,17).reshape((4,4)).astype(numpy.uint32)
#print(numpy.lib.stride_tricks.as_strided(A,shape=(2,2),strides=(16,4))) #4 means we have to skip 4 bytes for each column
                    #and 16 means we have to skip 16 bytes for each row
#print(numpy.lib.stride_tricks.as_strided(A,shape=(2,2),strides=(8,4))) #here,we skip 8 bytes,or 2 numbers, before starting the 
                    #creation of the next row
#print(numpy.lib.stride_tricks.as_strided(A,shape=(3,3,2,2),strides=(16,4,16,4))) #this returns every consecutive 3x3 window
#print(numpy.lib.stride_tricks.sliding_window_view(A,(3,3))) #returns every 3x3 consecutive window and does the same thing

#----------------------------------

A=numpy.array([1,6,2,7,3,4])
#print(A[numpy.argpartition(A,3)[:3]]) #this can be used to get nsmallest elements from the array
#print(A[numpy.argpartition(-A,3)[:3]]) #this can be used to get nlargest element from the array
    #argpartition basically sort the array only partially and returns the indexes of the nsmallest elements in the original array
    #in their corresponding positions

#----------------------------------

def cartesian(arrays):
    arrays=[numpy.asarray(arr) for arr in arrays]
    shape=[len(arr) for arr in arrays]
    ix=numpy.indices(shape)
    ix=ix.reshape(len(arrays),-1).T

    for n,arr in enumerate(arrays):
        ix[:,n]=arr[ix[:,n]]
    
    return ix 

#print(cartesian([[1,2,3],[4,5]])) #code to find the cartesian product of given arrays

#----------------------------------

A=numpy.array([[1,4,0],[2,3,5]])
#print(pow(A,4))
#print(A*A*A*A) #two ways of computing the power of a matrix

#----------------------------------


A=numpy.array([1,4,5,15,32,64,128,256])
B=(A.reshape(-1,1) & numpy.array([1<<i for i in range(len(A))])!=0).astype(int)
#print(B[:,::-1]) #converting the given numbers to binary representations in matrix

#----------------------------------

A = numpy.array([0,1,2])

B = numpy.array([[0,1,2,3],
              [4,5,6,7],
              [8,9,10,11]])

M1=(A[:,numpy.newaxis]*B) #matrix multiplication
M2=numpy.einsum('i,ij->ij',A,B) #einsum performs the same operation but doesn't need temporary storage and 
                                #is preferable especially in larger matrices
T1=M1.sum(axis=1) #calculating row-wise sums
T2=numpy.einsum('i,ij->i',A,B) #equivalent to T1 but more efficient
#we basically tell einsum that there is a dimension of A (i) and two of B(i,j). i,ij->ij means multiply the matrices
#and return dimension ij(basic multiplication). i,ij->i tells einsum that return only the dimension i(perform row-rise sum)

C=numpy.array([[1,0,4],[5,1,2]])
D=numpy.array([[3,1],[0,4],[5,2]])
P1=numpy.dot(C,D) #dot product
P2=numpy.einsum('ij,jk->ik',C,D) #this einsum function performs the same operation as that of dot funtion

arr=numpy.array([1,2,3,4,5])
#print(numpy.einsum('i->',arr)) #summing up all elements of a 1D array
#print(numpy.einsum('ij->',C)) #summing up all element of a 2D array
#print(numpy.einsum('ij,ij->ij',C,C)) #equivalent to C*C
#print(numpy.einsum('ij,jk->ik',C,D)) #equivalent to numpy.dot(C,D) but more efficient
#print(numpy.einsum('i,j->ij',arr,arr)) #equivalent to numpy.outer(C,D)
#print(numpy.einsum('i,i->',arr,arr)) #equivalen to numpy.inner(arr,arr) and arr.T * arr
#print(numpy.einsum('ij->ji',C)) #transpose
#print(numpy.einsum('ij->i',C)) #row-wise sum
#print(numpy.einsum('ij->j',C)) #column-wise sum

#---------------------------------
