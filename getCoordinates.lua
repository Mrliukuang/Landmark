--------------------------------------------------------------------------------
-- Convert coordinate data into .t7 file
-- Output: Tensor sized [N,10]
--      - N is the # of people
--      - 10 is the 5 landmark coordinates as [x1,y1,x2,y2,...,x5,y5]
--------------------------------------------------------------------------------

npy4th = require 'npy4th'

imPath = '/mnt/hgfs/D/download/AFLW/aflw/faces5/'

N = 11989

data = npy4th.loadnpy('../AFLW/scaledCoordinates.npy')
#data

data = data:reshape(N,10);

torch.type(data)
dataInt = data:int()

torch.save('coordinates.t7', dataInt)
