--------------------------------------------------------------------------------
-- Convert images file into .t7
--------------------------------------------------------------------------------
require 'image';

-- Load the images
imPath = '/search/data/user/liukuang/dataset/AFLW/faces5/'

N = 11989    -- N faces
--N = 10
L = 224      -- image resize to [L, L]
images = torch.IntTensor(N,3,L,L)

f = io.open('imNames.txt',"r")
while true do
    line = f:read('*l')
    if not line then break end

    i = (i or 0) + 1
    imName = line..'.jpg'
    print(imName, i, N)
    -- load image
    im = image.load(imPath..imName,3,'float')
    im = im*255
    -- resize
    im = image.scale(im,L,L)
    -- permuate
    im_bgr = im:index(1,torch.LongTensor{3,2,1})
    -- zero mean
    mean = {129.1863,104.7624,93.5940}
    for i = 1,3 do
        im_bgr[i]:add(-mean[i])
    end

    images[i] = im_bgr:int()
end
f:close()

torch.save('images.t7',images)

-- Forward the network to get features
