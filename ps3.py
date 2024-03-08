# %%
import numpy as np



# %%
Wf = np.array([0, 0])
bf = np.array([1])

Wo = np.array([0, 0])
bo = np.array([1])

Wi = np.array([0, 1])
bi = np.array([0])

Wc = np.array([-2, 1])
bc = np.array([0])

# %%
bit_string = [0,1,1,0,1,0]
parity = [0,1,0,0,1,1]


def sigmoid(x):
    return 1 if x > 0 else 0

def tanh(x):
    if x>0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1



def lstm(x, h0=0, c0=0):
    f = sigmoid(np.matmul(Wf, np.array([h0, x])) +bf)
    i = sigmoid(np.matmul(Wi, np.array([h0, x])) +bi)
    o = sigmoid(np.matmul(Wo, np.array([h0, x])) +bo)
    c_tilde = tanh(np.matmul(Wc, np.array([h0, x])) + bc)
    c = f*c0 + i*c_tilde
    h = o*tanh(c)
    return h,c,o
    

# %%
h0 = 0
c0 = 0

for xi in bit_string:
    h0,c0,o = lstm(xi, h0=h0, c0=c0)
    print(c0)
# %%


def running_parity(binary_array):
    running_parity_array = []
    parity = 0
    for num in binary_array:
        # Calculate parity for the current binary number
        parity ^= num
        running_parity_array.append(parity)
    return running_parity_array

# Generate an array of binary numbers
binary_array = [np.random.randint(0, 2) for _ in range(20)]
# Calculate the running parity for the binary numbers
running_parity_array = running_parity(binary_array)

parity = []
output = []
hidden = []
h0 = 0
c0 = 0
for xi in binary_array:
    h0,c0,o = lstm(xi, h0=h0, c0=c0)
    parity.append(c0)
    output.append(o)
    hidden.append(h0)
assert running_parity_array == parity

print(binary_array)
print(running_parity_array)
print(parity)
# %%
