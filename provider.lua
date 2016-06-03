local Provider = torch.class 'Provider'

function Provider:__init()
    local samples = torch.load('./features.t7')
    local targets = torch.load('./coordinates.t7')

    local N = samples:size(1)
    local trsize = math.ceil(N*0.8)
    local tesize = N-trsize

    self.trainData = {
        data = samples[{{1,trsize}}]:float(),
        labels = targets[{{1,trsize}}]:float(),
        size = function() return trsize end
    }

    self.testData = {
        data = samples[{{trsize+1,N}}]:float(),
        labels = targets[{{trsize+1,N}}]:float(),
        size = function() return tesize end
    }

	self:normalize()
end

function Provider:normalize()
    local trainMean = self.trainData.data:mean(1)
    local trainStd = self.trainData.data:std(1)

    local trainData = self.trainData.data
    trainData:add(-trainMean:expandAs(trainData)):cdiv(trainStd:expandAs(trainData))
    local testData = self.testData.data
    testData:add(-trainMean:expandAs(testData)):cdiv(trainStd:expandAs(testData))
	
    
    local labelMean = self.trainData.labels:mean(1)
--	local labelStd = self.trainData.labels:std(1)

    self.trainData.labels:add(-labelMean:expandAs(self.trainData.labels))
	self.testData.labels:add(-labelMean:expandAs(self.testData.labels))
end
