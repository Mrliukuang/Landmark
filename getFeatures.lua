require 'nn'
require 'cunn'

-- Load images
print('Loading images..')
images = torch.load('./images.t7')
N = images:size(1)

-- Load CNN model
print('Loading model..')
netPath = './VGG_FACE/VGG_FACE.t7'
net = torch.load(netPath):cuda()
net:evaluate()

--y = net:forward(images:float())

y = torch.CudaTensor(N,2622)

for i = 1,N do 
	print(i,N)
	im = images[i]
	y[i] = net:forward(im:cuda()):clone()
end

torch.save('features.t7', y)


-- Zero mean & Normalization
print('Normalizing..')
ymean = y:mean(1)
ystd = y:std(1)

yy = (y-ymean:expandAs(y)):cdiv(ystd:expandAs(y))

torch.save('features_normalized.t7', yy)

