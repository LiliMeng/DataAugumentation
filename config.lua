-- Make sure the local modules are visible to the code.
local thisPath        = paths.dirname(paths.thisfile())
if not doMain then
    paths.dofile('../../../../../set_env.lua')
end

-- package.path    = thisPath .. "/?.lua;" .. thisPath .. "/../../modules/?.lua;" .. package.path
-- package.cpath   = thisPath .. "/../../modules/?.so;"  .. package.cpath
require 'program'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nnx'
require 'optim'
require 'nngraph'

function getConfig()
    local config = {
        inputWidth       = 256,
        inputHeight      = 256,
        imageScaler      = 1/2,
        priorWidth       = 38.5,
        priorHeight      = 16.5,
        neighbourWidth   = 30,
        neighbourHeight  = 10,

        nThreads         = 3,
        batchSize        = 8,
        dataset          = 'XTGTData',
        dataset_path     = '/scratch/datasets/bb1/',

        displayInterval  = 20,
        testInterval     = 100,
        nTestDisplay     = 15,
        nTestIterations  = 3,


        -- trainBatchSize   = 128,
        -- valBatchSize     = 256,
        snapshotInterval = 1000,
        maxIterations    = 100000,
        optimMethod      = optim.adadelta,
        optimConfig      = {},
                            -- {
                            -- learningRate        = 1e-5,
                            -- learningRateDecay   = 0.0,
                            -- momentum            = 0.99,
                            -- weightDecay         = 0.0001,
                            -- },
        gpu              = 1,
    }
    return config
end

local detloss = torch.class('XTDetLoss')

function detloss:__init(config)
    self.config = config
    self.bceCrit = nn.BCECriterion():cuda()
    self.mseCrit = nn.MSECriterion():cuda()
end

function detloss:forward(input, target)
    -- torch.save('test.t7', {input=input, target=target})
    -- xtError('Doing forward')
    confidence = input
    local targetConfidence = confidence:clone():fill(0)

    self.predictions = {}
    local true_conf = 0
    local total_box = 0
    for n = 1, #target do
        local samplePreds = {}
        for b = 1, #target[n] do
            total_box = total_box + 1
            local coords = target[n][b].coords

            local bwidth = coords[4]-coords[2]+1
            local bx = math.floor(coords[2] + bwidth/2) -- center x
            local bheight = coords[3]-coords[1]+1
            local by = math.floor(coords[1] + bheight/2)

            targetConfidence[{{n},{1},{coords[1],coords[3]},{coords[2],coords[4]}}]:fill(1)
            -- table.insert(samplePreds, 
            --     {mergedCoordinates[{{n},{1,4},{by},{bx}}]:float():view(-1),
            --     confidence[n][1][by][bx]})
            true_conf = true_conf + confidence[n][1][by][bx]
        end
        table.insert(self.predictions, samplePreds)
    end
    self.true_conf = true_conf/total_box
    self.confidence = confidence
    self.targetConfidence = targetConfidence
    self.confidenceIm = confidence

    local l2loss = 0
    local bcloss = self.bceCrit:forward(confidence, targetConfidence)

    self.regLoss = l2loss
    self.conLoss = bcloss


    return bcloss
end

function detloss:backward(input, target)
    -- xtError('Doing backward')

    local bcgrad = self.bceCrit:backward(input, self.targetConfidence)
    
    return bcgrad
end

function createModel(config)

    require('cutorch')
    require('nn')
    require('cunn')
    require('cudnn')
    require('nngraph')
    require 'nnx'

    local shortcutType = 'B'

    local Convolution = cudnn.SpatialConvolution
    local Avg = cudnn.SpatialAveragePooling
    local ReLU = cudnn.ReLU
    local Max = cudnn.SpatialMaxPooling
    local SBatchNorm = cudnn.SpatialBatchNormalization

    -- The shortcut layer is either identity or 1x1 convolution
    local function shortcut(nInputPlane, nOutputPlane, strideX, strideY)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, strideX, strideY))
            :add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, strideX, strideY))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
    end

    -- The basic residual layer block for 18 and 34 layer network, and the
    -- CIFAR networks
    local function basicblock(n, strideX, strideY)
      local nInputPlane = iChannels
      iChannels = n 

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3,strideX,strideY,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, strideX, strideY)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
    end

    -- The original bottleneck residual layer for 50, 101, and 152 layer network
    local function resnet_bottleneck(n, stride)
        local nInputPlane = iChannels
        iChannels = n * 4

        local s = nn.Sequential()
        s:add(Convolution(nInputPlane, n, 1,1,1,1, 0, 0))
        s:add(SBatchNorm(n))
        s:add(ReLU(true))
        s:add(Convolution(n,n,3,3,stride,stride,1,1))
        s:add(SBatchNorm(n))
        s:add(ReLu(true))
        s:add(Convolution(n, n*4, 1, 1, 1, 1, 0, 0))
        s:add(SBatchNorm(n*4))

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(s)
                :add(shortcut(nInputPlane, n*4, stride)))
            :add(nn.CAddTable(true))
            :add(ReLU(true))
    end

    -- The aggregated residual transformation bottleneck layer, Form (B)
    local function split(nInputPlane, d, c, stride)
        local cat = nn.ConcatTable()
        for i=1, c do
            local s = nn.Sequential()
            s:add(Convolution(nInputPlane, d, 1, 1, 1,1, 0, 0))
            s:add(SBatchNorm(d))
            s:add(ReLU(true))
            s:add(Convolution(d,d,3,3,stride, stride, 1,1))
            s:add(SBatchNorm(d))
            s:add(ReLU(true))
            cat:add(s)
        end
        return cat
    end

    local function resnext_bottleneck_B(n, stride)
        local nInputPlane = iChannels
        iChannels = n *4

        local D = math.floor(n*(64/64))
        local C = 32

        local s = nn.Sequential()
        s:add(split(nInputPlane, D, C, stride))
        s:add(nn.JoinTable(2))
        s:add(Convolution(D*C, n*4, 1, 1, 1,1, 0,0))
        s:add(SBatchNorm(n*4))

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(s)
                :add(shortcut(nInputPlane, n*4, stride)))
            :add(nn.CAddTable(true))
            :add(ReLU(true))
    end 

    -- The aggregated residual transformation bottleneck layer, Form (C)
    local function resnext_bottleneck_C(n, stride)
        local nInputPlane = iChannels
        iChannels = n*4

        local D = math.floor(n*(64/64))
        local C = 32

        local s = nn.Sequential()
        s:add(Convolution(nInputPlane, D*C, 1, 1,1,1,0,0))
        s:add(SBatchNorm(D*C))
        s:add(ReLU(true))
        s:add(Convolution(D*C, D*C, 3, 3, stride, stride, 1, 1, C))
        s:add(SBatchNorm(D*C))
        s:add(ReLU(true))
        s:add(Convolution(D*C, n*4, 1,1,1,1,0,0))
        s:add(SBatchNorm(n*4))

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(s)
                :add(shortcut(nInputPlane, n*4, stride)))
            :add(nn.CAddTable(true))
            :add(ReLU(true))
    end 

    -- Creates count residual blocks with specified number of features
    local function layer(block, features, count, strideX, strideY)
        local s = nn.Sequential()
        for i=1,count do
            s:add(block(features, i == 1 and strideX or 1, i == 1 and strideY or 1))
        end
        return s
    end

    iChannels = 16

    local inp = nn.Identity()()

    bottleneck = resnext_bottleneck_B

    -- model and criterion
    local model_f1 = nn.Sequential()
    model_f1:add(nn.AddConstant(-128.0))
    model_f1:add(nn.MulConstant(1.0 / 128))

    model_f1:add(Convolution(1,16, 3,3, 1,1, 1,1))
    model_f1:add(SBatchNorm(16))
    model_f1:add(ReLU(true))
    model_f1:add(layer(basicblock, 16, 2))

    local f1 = model_f1(inp) -- 1x16x32x200

    local model_f2 = nn.Sequential()
    model_f2:add(layer(bottleneck, 32, 4, 2, 2))
    local f2 = model_f2(f1) -- 1x32x16x100

    local model_f3 = nn.Sequential()
    model_f3:add(layer(bottleneck, 64, 6, 2, 2))   
    local f3 = model_f3(f2)

    local model_f4 = nn.Sequential()
    model_f4:add(layer(bottleneck, 128, 8, 1, 2))   
    local f4 = model_f4(f3)

    local model_f5 = nn.Sequential()
    model_f5:add(layer(bottleneck, 256, 10, 1, 2))
    local f5 = model_f5(f4)

    local model_f6 = nn.Sequential()
    model_f6:add(layer(bottleneck, 512, 4, 1, 2))
    local f6 = model_f6(f5)

    local upf6 = nn.SpatialFullConvolution(512,256, 1,2, 1,2,0,0,0,0)(f6)
    local pf5  = Convolution(256,256, 1,1, 1,1, 0,0)(f5)
    local fuse_f5 = nn.CAddTable()({pf5, upf6})

    local upf5 = nn.SpatialFullConvolution(256, 256, 1,2, 1,2,0,0,0,0)(fuse_f5)
    local pf4  = Convolution(128,256, 1,1, 1,1, 0,0)(f4)
    local fuse_f4 = nn.CAddTable()({pf4, upf5})

    local upf4 = nn.SpatialFullConvolution(256,256, 1,2, 1,2,0,0,0,0)(fuse_f4)
    local pf3  = Convolution(64,256, 1,1, 1,1, 0,0)(f3)
    local fuse_f3 = nn.CAddTable()({pf3, upf4})

    local upf3 = nn.SpatialFullConvolution(256,256, 2,2, 2,2,0,0,0,0)(fuse_f3)
    local pf2  = Convolution(32,256, 1,1, 1,1, 0,0)(f2)
    local fuse_f2 = nn.CAddTable()({pf2, upf3})

    local upf2 = nn.SpatialFullConvolution(256,1, 2,2, 2,2,0,0,0,0)(fuse_f2)
    local seg_output = cudnn.Sigmoid()(upf2)

    local model = nn.gModule({inp},{seg_output})

    local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
    end
    local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
    end
    
    ConvInit('cudnn.SpatialConvolution')
    ConvInit('nn.SpatialConvolution')
    BNInit('fbnn.SpatialBatchNormalization')
    BNInit('cudnn.SpatialBatchNormalization')
    BNInit('nn.SpatialBatchNormalization')

    local g = model

    g:cuda()

    return g, detloss.new(config)
end


if doMain() then
    model = createModel(getConfig())
    output = model:forward(torch.randn(5,1, 256,256):cuda());
end
