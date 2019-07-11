import h5py
import numpy as np
import os
from scipy.spatial import cKDTree
from scipy import signal

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = "ignore", category = FutureWarning)

class badlandsOutput:

    def __init__(self,folder=None, timestep=0):
        """
        Initialization function.

        Parameters
        ----------
        variable : folder
            Folder path to Badlands outputs.

        variable : timestep
            Time step to load.
        """

        self.folder = folder+'/h5'
        self.timestep = timestep
        self.shaded = None
        self.Xarr = None
        self.Yarr = None
        self.xi = None
        self.yi = None
        self.z = None
        self.wavesed = None
        self.wavestress = None
        self.cumhill = None
        self.depreef = None
        self.dz = None
        self.dataExtent =  None
        self.dx = None

        self.slp = None
        self.aspect = None
        self.hcurv = None
        self.vcurv = None

        self.hydro = None

        self.regionx = None
        self.regiony = None
        self.regiondz = None
        self.regionz = None

        # Get points on the grid which are within the catchment
        self.concave_hull = None
        self.edge_points = None
        self.xpts = None
        self.ypts = None
        self.rdz = None

        return

    def regridTINdataSet(self, smth=2,dx=None):
        """
        Read the HDF5 file for a given time step and build slope and aspect

        Parameters
        ----------
        variable: smth
            Gaussian filter

        variable: dx
            Discretisation value in metres.

        """

        azimuth=315.0
        altitude=45.0

        if not os.path.isdir(self.folder):
            raise RuntimeError('The given folder cannot be found or the path is incomplete.')

        df = h5py.File('%s/tin.time%s.p%s.hdf5'%(self.folder, self.timestep, 0), 'r')
        coords = np.array((df['/coords']))
        cumdiff = np.array((df['/cumdiff']))
        wavesed = np.array((df['/cumwave']))
        x, y, z = np.hsplit(coords, 3)
        wavestress = np.array((df['/waveS']))
        cumhill = np.array((df['/cumhill']))
        reef1 = np.array((df['/depSpecies1']))
        reef2 = np.array((df['/depSpecies2']))

        self.dx = dx
        if dx is None:
            self.dx = (x[1]-x[0])[0]
            #print 'Set dx to:',dx
        dx = self.dx
        nx = int((x.max() - x.min())/dx+1)
        ny = int((y.max() - y.min())/dx+1)
        xi = np.linspace(x.min(), x.max(), nx)
        yi = np.linspace(y.min(), y.max(), ny)
        self.Xarr = xi
        self.Yarr = yi

        xi, yi = np.meshgrid(xi, yi)
        xyi = np.dstack([xi.flatten(), yi.flatten()])[0]
        XY = np.column_stack((x,y))
        tree = cKDTree(XY)
        distances, indices = tree.query(xyi, k=3)
        z_vals = z[indices][:,:,0]
        zi = np.average(z_vals,weights=(1./distances), axis=1)

        dz_vals = cumdiff[indices][:,:,0]
        dzi = np.average(dz_vals,weights=(1./distances), axis=1)

        ws_vals = wavesed[indices][:,:,0]
        wsi = np.average(ws_vals,weights=(1./distances), axis=1)

        cumhill_vals = cumhill[indices][:,:,0]
        hilli = np.average(cumhill_vals,weights=(1./distances), axis=1)

        ss_vals = wavestress[indices][:,:,0]
        ssi = np.average(ss_vals,weights=(1./distances), axis=1)

        r1_vals = reef1[indices][:,:,0]
        r1i = np.average(r1_vals,weights=(1./distances), axis=1)

        r2_vals = reef2[indices][:,:,0]
        r2i = np.average(r2_vals,weights=(1./distances), axis=1)

        onIDs = np.where(distances[:,0] == 0)[0]
        if len(onIDs) > 0:
            zi[onIDs] = z[indices[onIDs,0],0]
            dzi[onIDs] = cumdiff[indices[onIDs,0],0]
            wsi[onIDs] = wavesed[indices[onIDs,0],0]
            hilli[onIDs] = cumhill[indices[onIDs,0],0]
            ssi[onIDs] = wavestress[indices[onIDs,0],0]
            r1i[onIDs] = reef1[indices[onIDs,0],0]
            r2i[onIDs] = reef2[indices[onIDs,0],0]

        z = np.reshape(zi,(ny,nx))
        dz = np.reshape(dzi,(ny,nx))
        ws = np.reshape(wsi,(ny,nx))

        hs = np.reshape(hilli,(ny,nx))
        ss = np.reshape(ssi,(ny,nx))
        r1 = np.reshape(r1i,(ny,nx))
        r2 = np.reshape(r2i,(ny,nx))

        reef = r1+r2
        reef[reef>0] = 1

        # Calculate gradient
        Sx, Sy = np.gradient(z)

        rad2deg = 180.0 / np.pi
        slope = 90. - np.arctan(np.sqrt(Sx**2 + Sy**2))*rad2deg
        slp = np.sqrt(Sx**2 + Sy**2)

        aspect = np.arctan2(-Sx, Sy)
        deg2rad = np.pi / 180.0
        shaded = np.sin(altitude*deg2rad) * np.sin(slope*deg2rad) \
                 + np.cos(altitude*deg2rad) * np.cos(slope*deg2rad) \
                 * np.cos((azimuth - 90.0)*deg2rad - aspect)

        shaded = shaded * 255

        self.shaded = shaded
        self.xi = xi
        self.yi = yi
        self.z = z
        self.dz = dz
        self.wavesed = ws
        self.wavestress = ss
        self.cumhill = hs
        self.depreef = reef
        self.dataExtent = [np.amin(xi), np.amax(xi), np.amin(yi), np.amax(yi)]

        # Applying a Gaussian filter
        self.cmptParams(xi,yi,z)
        z_gauss = self.smoothData(z, smth)
        dz_gauss = self.smoothData(dz, smth)
        self.cmptParams(xi, yi, z_gauss)

        return

    def cmptParams(self,x,y,Z):
        """
        Define aspect, gradient and horizontal/vertical curvature using a
        quadratic polynomial method.
        """

        # Assign boundary conditions
        Zbc = self.assignBCs(Z,x.shape[0],x.shape[1])

        # Neighborhood definition
        # z1     z2     z3
        # z4     z5     z6
        # z7     z8     z9

        z1 = Zbc[2:, :-2]
        z2 = Zbc[2:,1:-1]
        z3 = Zbc[2:,2:]
        z4 = Zbc[1:-1, :-2]
        z5 = Zbc[1:-1,1:-1]
        z6 = Zbc[1:-1, 2:]
        z7 = Zbc[:-2, :-2]
        z8 = Zbc[:-2, 1:-1]
        z9 = Zbc[:-2, 2:]

        # Compute coefficient values
        dx = x[0,1]-x[0,0]
        zz = z2+z5
        r = ((z1+z3+z4+z6+z7+z9)-2.*(z2+z5+z8))/(3. * dx**2)
        t = ((z1+z2+z3+z7+z8+z9)-2.*(z4+z5+z6))/(3. * dx**2)
        s = (z3+z7-z1-z9)/(4. * dx**2)
        p = (z3+z6+z9-z1-z4-z7)/(6.*dx)
        q = (z1+z2+z3-z7-z8-z9)/(6.*dx)
        u = (5.*z1+2.*(z2+z4+z6+z8)-z1-z3-z7-z9)/9.
        #
        with np.errstate(invalid='ignore',divide='ignore'):
            grad = np.arctan(np.sqrt(p**2+q**2))
            aspect = np.arctan(q/p)
            hcurv = -(r*q**2-2.*p*q*s+t*p**2) / \
                    ((p**2+q**2)*np.sqrt(1+p**2+q**2))
            vcurv = -(r*p**2+2.*p*q*s+t*q**2) /  \
                    ((p**2+q**2)*np.sqrt(1+p**2+q**2))

            self.slp = grad
            self.aspect = aspect
            self.hcurv = hcurv
            self.vcurv = vcurv

            return

    def assignBCs(self,z,nx,ny):
        """
        Pads the boundaries of a grid. Boundary condition pads the boundaries
        with equivalent values to the data margins, e.g. x[-1,1] = x[1,1].
        It creates a grid 2 rows and 2 columns larger than the input.
        """
        Zbc = np.zeros((nx + 2, ny + 2))
        Zbc[1:-1,1:-1] = z

        # Assign boundary conditions - sides
        Zbc[0, 1:-1] = z[0, :]
        Zbc[-1, 1:-1] = z[-1, :]
        Zbc[1:-1, 0] = z[:, 0]
        Zbc[1:-1, -1] = z[:,-1]

        # Assign boundary conditions - corners
        Zbc[0, 0] = z[0, 0]
        Zbc[0, -1] = z[0, -1]
        Zbc[-1, 0] = z[-1, 0]
        Zbc[-1, -1] = z[-1, 0]

        return Zbc

    def gaussianFilter(self,sizex,sizey=None,scale=0.333):
        '''
        Generate and return a 2D Gaussian function
        of dimensions (sizex,sizey)

        If sizey is not set, it defaults to sizex
        A scale can be defined to widen the function (default = 0.333)
        '''
        sizey = sizey or sizex
        x, y = np.mgrid[-sizex:sizex+1, -sizey:sizey+1]
        g = np.exp(-scale*(x**2/float(sizex)+y**2/float(sizey)))

        return g/g.sum()

    def smoothData(self,dem, smth=2):
        '''
        Calculate the slope and gradient of a DEM
        '''

        gaussZ = np.zeros((dem.shape[0]+6,dem.shape[1]+6))
        gaussZ[3:-3,3:-3] = dem

        f0 = self.gaussianFilter(smth)
        smoothDEM = signal.convolve(gaussZ,f0,mode='valid')

        return smoothDEM[1:-1,1:-1]
