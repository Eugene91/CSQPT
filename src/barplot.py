import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import os
import numpy as np


def arrange_plot(data, ax, label):
    
    lx = len(data[0])            # Work out matrix dimensions
    ly = len(data[:,0])

    column_names = lx
    row_names = ly

    xpos = np.arange(0,lx,1)    # Set up a mesh of positions
    ypos = np.arange(0,ly,1)
    xpos, ypos = np.meshgrid(xpos, ypos)

    xpos = xpos.flatten()   # Convert positions to 1D array
    ypos = ypos.flatten()
    zpos = np.zeros(lx*ly)

    dx = 0.5*np.ones_like(zpos)
    dy = dx.copy()
    dz = data.flatten()


    ax.bar3d(xpos,ypos,zpos, dx, dy, dz ) #color=cs

    ax.xaxis.set_major_locator(ticker.FixedLocator((xpos+0.25)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter((xpos)))

    ax.yaxis.set_major_locator(ticker.FixedLocator((ypos+0.25)))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter((ypos)))


    ax.w_yaxis.set_ticklabels(ypos)
    ax.set_xlabel('n')
    ax.set_ylabel('m')
    ax.set_zlim(-0.5, 1)
    ax.set_zlabel(label)
    


def bar_plot(rho, folderName='Images/', name='plot', label="$\\rho_{nm}$", plot_img_part = True):
    
    data = np.real(rho)


    fig = plt.figure(figsize=plt.figaspect(0.4))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    
    arrange_plot(data, ax, label = f"Re {label}")
    
    
    
    
    if plot_img_part:
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        data = np.imag(rho)
        arrange_plot(data, ax, label = f"Im {label}")

    

    
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    
    plt.savefig(f'{folderName}/{name}.png')



# def bar_plot(rho, folderName='Images/', name='plot', plot_img_part = True):
    
#     data = np.real(rho)


#     fig = plt.figure(figsize=plt.figaspect(0.4))

#     ax = fig.add_subplot(1, 2, 1, projection='3d')
    

#     lx = len(data[0])            # Work out matrix dimensions
#     ly = len(data[:,0])

#     column_names = lx
#     row_names = ly



#     xpos = np.arange(0,lx,1)    # Set up a mesh of positions
#     ypos = np.arange(0,ly,1)
#     xpos, ypos = np.meshgrid(xpos, ypos)

#     xpos = xpos.flatten()   # Convert positions to 1D array
#     ypos = ypos.flatten()
#     zpos = np.zeros(lx*ly)

#     dx = 0.5*np.ones_like(zpos)
#     dy = dx.copy()
#     dz = data.flatten()


#     ax.bar3d(xpos,ypos,zpos, dx, dy, dz ) #color=cs

#     ax.xaxis.set_major_locator(ticker.FixedLocator((xpos+0.25)))
#     ax.xaxis.set_major_formatter(ticker.FixedFormatter((xpos)))

#     ax.yaxis.set_major_locator(ticker.FixedLocator((ypos+0.25)))
#     ax.yaxis.set_major_formatter(ticker.FixedFormatter((ypos)))


#     ax.w_yaxis.set_ticklabels(ypos)
#     ax.set_xlabel('n')
#     ax.set_ylabel('m')
#     ax.set_zlim(-0.5, 1)
#     ax.set_zlabel("Re $\\rho_{nm}$")

#     ax = fig.add_subplot(1, 2, 2, projection='3d')
    
#     if plot_img_part:

#     data = np.imag(rho)

#     lx = len(data[0])            # Work out matrix dimensions
#     ly = len(data[:,0])

#     column_names = lx
#     row_names = ly



#     xpos = np.arange(0,lx,1)    # Set up a mesh of positions
#     ypos = np.arange(0,ly,1)
#     xpos, ypos = np.meshgrid(xpos, ypos)

#     xpos = xpos.flatten()   # Convert positions to 1D array
#     ypos = ypos.flatten()
#     zpos = np.zeros(lx*ly)

#     dx = 0.5*np.ones_like(zpos)
#     dy = dx.copy()
#     dz = data.flatten()


#     ax.bar3d(xpos,ypos,zpos, dx, dy, dz ) #color=cs

#     ax.xaxis.set_major_locator(ticker.FixedLocator((xpos+0.25)))
#     ax.xaxis.set_major_formatter(ticker.FixedFormatter((xpos)))

#     ax.yaxis.set_major_locator(ticker.FixedLocator((ypos+0.25)))
#     ax.yaxis.set_major_formatter(ticker.FixedFormatter((ypos)))


#     ax.w_yaxis.set_ticklabels(ypos)
#     ax.set_xlabel('n')
#     ax.set_ylabel('m')
#     ax.set_zlim(-0.5, 1)
#     ax.set_zlabel("Im $\\rho_{nm}$")
    
#     if not os.path.exists(folderName):
#         os.makedirs(folderName)
    
#     plt.savefig(f'{folderName}/{name}.png')
    
    
def diagElem(X,m,n,Hdims):
    DT=diagProj(Hdims,m,n)
    Xp=np.dot(DT,np.dot(X,DT))
    return (Xp[Xp!=0][0])

def getDiagTensor(X,Hdims):
    tensor = np.zeros((Hdims,Hdims))
    for i in np.arange(0,Hdims):
        for k in np.arange(0,Hdims):
            tensor[i,k]=np.real(diagElem(X=X,m=i,n=k,Hdims=Hdims))
    return tensor       

def plotTensor(name,data):
    fig = plt.figure(figsize=plt.figaspect(0.35)) # figsize=(8, 10)

    ax = fig.add_subplot(1, 2, 1, projection='3d')


    lx = len(data[0])            # Work out matrix dimensions
    ly = len(data[:,0])
    print(lx)
    print(ly)

    column_names =  ['00,01,10,11'] #;lx
    row_names = ly



    xpos =np.arange(0,lx,1)    # Set up a mesh of positions
    ypos = np.arange(0,ly,1)
    xpos, ypos = np.meshgrid(xpos, ypos)

    xpos = xpos.flatten()   # Convert positions to 1D array
    ypos = ypos.flatten()
    zpos = np.zeros(lx*ly)

    dx = 0.4*np.ones_like(zpos)
    dy = 0.25*np.ones_like(zpos)
    dz = data.flatten()
    A=np.zeros((16,4))
    A[0:4] = [[0.9372549 , 0.2627451 , 0.24705882,1], 
       [0.9372549 , 0.2627451 , 0.24705882,1],
       [0.9372549 , 0.2627451 , 0.24705882,1],
       [0.9372549 , 0.2627451 , 0.24705882,1]]
    
    A[4:8] = [[0., 0.43137255, 0.23921569,1],
       [0., 0.43137255, 0.23921569,1],
       [0., 0.43137255, 0.23921569,1],
       [0., 0.43137255, 0.23921569,1]]
    
    A[8:12] = [[0.13333333, 0.25490196, 0.61176471, 1],
       [0.13333333, 0.25490196, 0.61176471, 1],
       [0.13333333, 0.25490196, 0.61176471, 1],
       [0.13333333, 0.25490196, 0.61176471, 1]]
    #colors = np.array(['darkred','darkgreen','dodgerblue','gold','dodgerblue','gold'])
    
    colors = np.array(['#ef433f','#006e3d','#22419c','black','dodgerblue','gold'])
    #colors =  plt.cm.jet(data.flatten()/float(data.max()))
    
    print(colors)
    ax.bar3d(xpos,ypos,zpos, dx, dy, dz,alpha=0.6,color=A) #color=cs color=colors[i]
        #ax.bar(xpos[i],ypos[i],zpos[i], zdir='y', alpha=0.8, color=colors[i])
        

    ax.xaxis.set_major_locator(ticker.FixedLocator((xpos+0.25)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter((xpos)))

    ax.yaxis.set_major_locator(ticker.FixedLocator((ypos+0.2)))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter((ypos)))


    ax.w_yaxis.set_ticklabels(ypos)
    
    ax.set_title(name)
    ax.set_xlabel('n (OUT)')
    ax.set_ylabel('m (IN)')
    ax.set_zlim(-0, 1)
    ax.set_zlabel("$\\mathcal{E}^{mm}_{nn}$")
    angle = -50
    ax.view_init(30, angle)
    plt.savefig(f'{folderName}/{name}-diag-Process-tensor.pdf',bbox_inches='tight')