#import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from datetime import datetime
from sklearn.linear_model import LinearRegression

#read inputs from input.gacode
def readInput(qlistname, inputfile='0.5/input.gacode', qlen=41):
    line_nums = []
    with open(inputfile) as myFile:
        for num, line in enumerate(myFile, 1):
            for lookup in qlistname:
                if lookup in line:
                    #print(lookup+' found at line:', num)
                    line_nums.append(num)

    q = []
    for lnum in line_nums:
        lines = list(range(lnum+1,lnum+qlen+1))
        result = []
        with open(inputfile) as myFile:
            for num, line in enumerate(myFile, 1):
                if num in lines:
                    val = float(line.split()[1])
                    result.append(val)
        q.append(result)
        
    return np.array(q)

#calculate the first derivatives of rmaj, zmag, kappa, delta, zeta, 
#which will be used in the calculation of Jacobian
def derivative(r, s):
    dsdr = np.zeros(r.shape)
    dsdr[1:-1] = (s[2:]-s[:-2])/(r[2:]-r[:-2])
    dsdr[0] = (s[1]-s[0])/(r[1]-r[0])
    dsdr[-1] = (s[-1]-s[-2])/(r[-1]-r[-2])
    return dsdr


dir_workspace = '0.5/'
inputfile = dir_workspace+'input.gacode'
#list of quantites
qlistname = ['rho', 'rmin', 'rmaj', 'zmag', 'kappa', 'delta', 'zeta', '# ne | 10^19/m^3', '# te | keV']

qlist = readInput(qlistname, inputfile=inputfile)
q0, q1 = qlist.shape[0], qlist.shape[1]

qlist_ext = np.zeros([q0, 3*q1])
qlist_ext[:,:q1] = qlist
drho = qlist[0,1]-qlist[0,0]
reg = LinearRegression().fit(qlist[0,:].reshape(-1,1), qlist[1,:].reshape(-1,))
qlist_ext[0,:] = np.linspace(0,drho*(3*q1-1),num=3*q1)
qlist_ext[1,q1:] = reg.predict(qlist_ext[0,q1:].reshape(-1,1))
# pcoefs = np.polyfit(qlist[0,:].reshape(-1,), qlist[1,:].reshape(-1,), 2)
# qlist_ext[1,q1:] = np.polyval(pcoefs, qlist_ext[0,q1:].reshape(-1,))

for i in range(2,q0-2):
    qlist_ext[i, q1:] = qlist[i,-1]
qlist =qlist_ext

for j in range(2,7):
    qlist = np.vstack((qlist,derivative(qlist[1,:], qlist[j,:])))

np.savetxt("qlist.csv", qlist, delimiter=",")

print(f'qlist building completed! The shape of qlist is {qlist.shape}')

# plt.plot(qlist[0,:], qlist[1,:], '.')
# plt.show()

#important to notice: rho and r are not exactly linearly related, the interpolations should be done either on rho or r, not mixed.