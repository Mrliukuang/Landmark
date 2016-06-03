require 'nn'
require 'xlua'
require 'image'
require 'optim'
require './provider.lua'

c = require 'trepl.colorize'

opt = lapp[[
    -g,--gpu               (default 2)                   GPU ID
    -c,--checkpointPath    (default './checkpoints/')    checkpoint saving path
    -b,--batchSize         (default 128)                 batch size
    -r,--resume                                          resume from checkpoint
    -t,--type              (default cuda)                datatype: float/cuda
    ]]

if opt.type == 'cuda' then
    require 'cunn'
    --require 'cudnn'
    require 'cutorch'
    cutorch.setDevice(opt.gpu)
end

function cast(m)
    if opt.type == 'float' then
        return m:float()
    elseif opt.type == 'cuda' then
        return m:cuda()
    else
        error('Unknown data type: '..opt.type)
    end
end

-- Set up model
print(c.blue '==> '..'setting up model..')
net = nn.Sequential()
nInputs = 2622
nOutputs = 10
net:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
net:add(cast(nn.Linear(nInputs, nOutputs)))

parameters, gradParameters = net:getParameters()
criterion = cast(nn.MSECriterion())

-- Load data
print(c.blue '==> '..'loading data..')
--provider = Provider()
provider = torch.load('./provider3.t7')
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

-- Set up optimizer
print(c.blue '==> '..'configure optimizer..\n')
optimState = optimState or {
    learningRate = 0.000001,
    learningRateDecay = 0,--1e-7,
    weightDecay = 0,--0.0005,
    momentum = 0.9,
    nesterov = true,
    dampening = 0.0
    }


function train()
    net:training()
    epoch = (epoch or 0)+1

    targets = cast(torch.FloatTensor(opt.batchSize, 10))

    indices = torch.randperm(provider.trainData:size(1)):long():split(opt.batchSize)
    indices[#indices] = nil

    trainLoss = 0
    for k, v in pairs(indices) do
        inputs = provider.trainData.data:index(1,v)    -- [N, C, H, W]
        targets:copy(provider.trainData.labels:index(1,v))

        feval = function(x)
            if x~= parameters then
                parameters:copy(x)
            end
            gradParameters:zero()

            local outputs = net:forward(inputs)
            local f = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            net:backward(inputs, df_do)

            trainLoss = trainLoss + f

            return f, gradParameters
        end
        optim.sgd(feval, parameters, optimState)
    end

	trainLoss = trainLoss/#indices
    --print(c.Green '==> '..loss/#indices)
end

function test()
    net:evaluate()

    testLoss = 0
    local bs = 90
	local nBatch = 0
    for i = 1, provider.testData.data:size(1)-bs, bs do
        local outputs = net:forward(provider.testData.data:narrow(1,i,bs))
        local f = criterion:forward(outputs, cast(provider.testData.labels:narrow(1,i,bs)))
        testLoss = testLoss + f
		nBatch = nBatch+1
    end
    print(c.Green '==> '..trainLoss..'\t'..testLoss/nBatch)
end

-- do for 500 epochs
while true do
    train()
	test()

	if epoch%50==0 then
		torch.save(epoch..'.t7', net)
	end

end
