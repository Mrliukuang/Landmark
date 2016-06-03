require 'nn';
require 'image';


-- Load input coordinates
inputs = torch.load('inputs.t7')    -- [N,10]
#inputs
-- Load CNN model
netPath = '/mnt/hgfs/D/download/vgg_face_torch/VGG_FACE.t7'
net = torch.load(netPath)
net:evaluate()


-- Load image
im = image.load('/home/luke/mm.jpg',3,'float')
im = im*255
-- Resize
im = image.scale(im, 256, 256)
#im






-- Permute
im_bgr = im:index(1,torch.LongTensor{3,2,1})
#im_bgr

-- Zero mean
mean = {129.1863,104.7624,93.5940}
for i = 1,3 do
    im_bgr[i]:add(-mean[i])
end

-- Forward
prob = net(im_bgr)
#prob

prob


maxval,maxid = prob:max(1)
print(maxid)
