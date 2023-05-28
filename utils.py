import matplotlib.pyplot as plt
import numpy as np
import time

def dual_dna(gstr):
    out = []
    for x in gstr:
        if x == 'A':
            out.append('T')
        elif x == 'T':
            out.append('A')
        elif x == 'G':
            out.append('C')
        elif x == 'C':
            out.append('G')
    return out

def one_hot_encoding_dna(sequence):
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    encoding = []
    
    for base in sequence:
        encoding.append(mapping[base])
    
    return np.array(encoding)


def visualize_kernel(akernel,skernel,name):
    #print(akernel,skernel)
    #print(akernel.shape,skernel.shape)
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize = (11,11))
    # Plot the first complex matrix
    im1 = axs[0,0].imshow(np.angle(akernel.real), cmap='Blues')
    axs[0,0].set_title('Asymmetric Matrix (Re)')

    im1 = axs[0,1].imshow(np.angle(akernel.imag), cmap='Blues')
    axs[0,1].set_title('Asymmetric Matrix (Im)')

    # Plot the second complex matrix
    im2 = axs[1,0].imshow(np.angle(skernel.real), cmap='Reds')
    axs[1,0].set_title('Symmetric Matrix (Re)')

    im2 = axs[1,1].imshow(np.angle(skernel.imag), cmap='Reds')
    axs[1,1].set_title('Symmetric Matrix (Im)')
    # Add colorbars to the subplots
    fig.colorbar(im1, ax=axs[0])
    fig.colorbar(im2, ax=axs[1])
    # Display the figure
    #plt.show()
    plt.savefig('{}_{}.jpg'.format(name,time.time()), dpi = 600)