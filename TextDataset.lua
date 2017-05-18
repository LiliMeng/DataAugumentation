-- Image transforms for data augmentation and input normalization
-- Lili@Xtract May 16th

--local t = require 'transforms'
require 'image'

local image = require 'image'
local display = require 'display'



local M = {}
local TextDataset = torch.class('resNxt.TextDataset', M)

function TextDataset:__init()
  --self.im = image.load('../../../../bb1/2017-05-16.00-11-06/00ae6eff3d9ef9c2e0675a6e6411069e/4/00ae6eff3d9ef9c2e0675a6e6411069e-4.png')
  --self.tmp_im = im:clone()
  --self.height, self.width = im:size(2), im:size(3)
  --print(height)
  --print(width)
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function TextDataset.Compose(input, transforms)
  local tmp_composed_img = input:clone()
  for _, transform in ipairs(transforms) do
    tmp_composed_img = transform(tmp_composed_img)
  end
  return tmp_composed_img 
end


function TextDataset.ColorNormalize(input, meanstd)
      local tmp_im_1 = input:clone()
      for i=1,3 do
         tmp_im_1:add(-meanstd.mean[i])
         tmp_im_1:div(meanstd.std[i])
      end
      return tmp_im_1
end

-- Scales the smaller edge to size
function TextDataset.Scale(input, size, interpolation)
   local tmp_im_2=input:clone()
   interpolation = interpolation or 'bicubic'
      local w, h = tmp_im_2:size(3), tmp_im_2:size(2)
      if (w <= h and w == size) or (h <= w and h == size) then
         return tmp_im_2
      end
      if w < h then
         return image.scale(tmp_im_2, size, h/w * size, interpolation)
      else
         return image.scale(tmp_im_2, w/h * size, size, interpolation)
      end
end

-- Crop to centered rectangle
function TextDataset.CenterCrop(input, size)
      local w1 = math.ceil((input:size(3) - size)/2)
      local h1 = math.ceil((input:size(2) - size)/2)
      return image.crop(input, w1, h1, w1 + size, h1 + size) -- center patch
end

-- Random crop form larger image with optional zero padding
function TextDataset.RandomCrop(input, size, padding)
   padding = padding or 0
      if padding > 0 then
         local temp = input.new(3, input:size(2) + 2*padding, input:size(3) + 2*padding)
         temp:zero()
            :narrow(2, padding+1, input:size(2))
            :narrow(3, padding+1, input:size(3))
            :copy(input)
         input = temp
      end

      local w, h = input:size(3), input:size(2)
      if w == size and h == size then
         return input
      end

      local x1, y1 = torch.random(0, w - size), torch.random(0, h - size)
      local out = image.crop(input, x1, y1, x1 + size, y1 + size)
      assert(out:size(2) == size and out:size(3) == size, 'wrong crop size')
      return out
end

-- Four corner patches and center crop from image and its horizontal reflection
function TextDataset.TenCrop(input,size)

   local centerCrop = M.CenterCrop(size)
      local w, h = input:size(3), input:size(2)

      local output = {}
      for _, img in ipairs{input, image.hflip(input)} do
         table.insert(output, centerCrop(img))
         table.insert(output, image.crop(img, 0, 0, size, size))
         table.insert(output, image.crop(img, w-size, 0, w, size))
         table.insert(output, image.crop(img, 0, h-size, size, h))
         table.insert(output, image.crop(img, w-size, h-size, w, h))
      end

      -- View as mini-batch
      for i, img in ipairs(output) do
         output[i] = img:view(1, img:size(1), img:size(2), img:size(3))
      end

      return input.cat(output, 1)
end

-- Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style)
function TextDataset.RandomScale(input, minSize, maxSize)
      local w, h = input:size(3), input:size(2)

      local targetSz = torch.random(minSize, maxSize)
      local targetW, targetH = targetSz, targetSz
      if w < h then
         targetH = torch.round(h / w * targetW)
      else
         targetW = torch.round(w / h * targetH)
      end

      return image.scale(input, targetW, targetH, 'bicubic')
end


-- Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
function TextDataset.RandomSizedCrop(input, size)
   local scale = TextDataset.Scale(input,size)
   local crop = TextDataset.CenterCrop(input,size)

      local attempt = 0
      repeat
         local area = input:size(2) * input:size(3)
         local targetArea = torch.uniform(0.08, 1.0) * area

         local aspectRatio = torch.uniform(3/4, 4/3)
         local w = torch.round(math.sqrt(targetArea * aspectRatio))
         local h = torch.round(math.sqrt(targetArea / aspectRatio))

         if torch.uniform() < 0.5 then
            w, h = h, w
         end

         if h <= input:size(2) and w <= input:size(3) then
            local y1 = torch.random(0, input:size(2) - h)
            local x1 = torch.random(0, input:size(3) - w)

            local out = image.crop(input, x1, y1, x1 + w, y1 + h)
            assert(out:size(2) == h and out:size(3) == w, 'wrong crop size')

            return image.scale(out, size, size, 'bicubic')
         end
         attempt = attempt + 1
      until attempt >= 10

      -- fallback
      return crop(scale(input))
end

function TextDataset.HorizontalFlip(input, prob)
      local tmp_im_horizontal = input:clone()
      if torch.uniform() < prob then
         tmp_im_horizontal = image.hflip(tmp_im_horizontal)
      end
      return tmp_im_horizontal
end

function TextDataset.Rotation(input,deg)
      local tmp_im_rotate = input:clone()
      if deg ~= 0 then
         tmp_im_rotate = image.rotate(tmp_im_rotate, (torch.uniform() - 0.5) * deg * math.pi / 180, 'bilinear')
      end
      return tmp_im_rotate
end

-- Lighting noise (AlexNet-style PCA-based noise)
function TextDataset.Lighting(input, alphastd, eigval, eigvec)
      local tmp_im_lighting = input:clone()
      if alphastd == 0 then
         return tmp_im_lighting
      end

      local alpha = torch.Tensor(3):normal(0, alphastd)
      local rgb = eigvec:clone()
         :cmul(alpha:view(1, 3):expand(3, 3))
         :cmul(eigval:view(1, 3):expand(3, 3))
         :sum(2)
         :squeeze()

     
      for i=1,3 do
         tmp_im_lighting[i]:add(rgb[i])
      end
      return tmp_im_lighting
end



local function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
   dst:resizeAs(img)
   dst[1]:zero()
   dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
   dst[2]:copy(dst[1])
   dst[3]:copy(dst[1])
   return dst
end

function TextDataset.Saturation(input,var)
   local gs
      gs = gs or input.new()
      grayscale(gs, input)

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
end


function TextDataset.Brightness(input,var)
  
      local gs

      gs = gs or input.new()
      gs:resizeAs(input):zero()

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
end


function TextDataset.Contrast(input, var)
    local gs
    gs = gs or input.new()
    grayscale(gs, input)
    gs:fill(gs[1]:mean())

    local alpha = 1.0 + torch.uniform(-var, var)
    blend(input, gs, alpha)
    return input 
end

function TextDataset.RandomOrder(input, ts)
      local img = input.img or input 
      --local img = tmp_im_randomOrder.img or tmp_im_randomOrder
      local order = torch.randperm(#ts)
    
      for i=1,#ts do
         print('it worked 1')
         img = ts[order[i]]
         print('it worked 2')
         local filename_randomOrder="augmented_data/randomOrder"..i..".jpg"
         image.save(filename_randomOrder, img)
      end
      return img
end

function TextDataset.ColorJitter(input, brightness, contrast, saturation)
  --input = input:long()
  local brightness = brightness or 0
   local contrast = contrast or 0
   local saturation = saturation or 0
   local ts = {}
   if brightness ~= 0 then
      table.insert(ts, TextDataset.Brightness(input,brightness))
   end
   if contrast ~= 0 then
      table.insert(ts, TextDataset.Contrast(input,contrast))
   end
   if saturation ~= 0 then
      table.insert(ts, TextDataset.Saturation(input,saturation))
   end
   print(ts)
   if #ts == 0 then
      return input 
   end
   
   return TextDataset.RandomOrder(input,ts)
end


function TextDataset:preprocess()
    local im = image.load('../../../../bb1/2017-05-16.00-11-06/00ae6eff3d9ef9c2e0675a6e6411069e/4/00ae6eff3d9ef9c2e0675a6e6411069e-4.png')
    local tmp_im = im:clone()
    height, width = im:size(2), im:size(3)
    local filename_left="augmented_data/Image_00185_color_jitter_img.jpg"
    print(height)
    print(width)
    
    local colorNormlize_img = TextDataset.ColorNormalize(tmp_im, meanstd)
    local scaled_img = TextDataset.Scale(tmp_im,256)
    local centerCrop_img = TextDataset.CenterCrop(tmp_im, 256)
    local randomSizedCrop_img = TextDataset.RandomSizedCrop(tmp_im, 224)
    --local tenCrop_img = TextDataset.TenCrop(tmp_im,256)
    local random_scale_img = TextDataset.RandomScale(tmp_im,20,1000)
    local horizontal_flip_img = TextDataset.HorizontalFlip(tmp_im, 0.5)
    local rotated_img = TextDataset.Rotation(tmp_im,20) 
    local brightness_img =TextDataset.Brightness(tmp_im, 12)
    local lighting_img = TextDataset.Lighting(tmp_im, 0.1, pca.eigval, pca.eigvec)
    local saturation_img = TextDataset.Saturation(tmp_im, 15)
    local constrasted_img = TextDataset.Contrast(tmp_im, 100)
    --local randomOrder_img = TextDataset.RandomOrder(tmp_im, )
   -- local ts = {}
    -- table.insert(ts, TextDataset.Brightness(tmp_im,12))
    
    local color_jitter_img = TextDataset.ColorJitter(tmp_im, 100,  0.1, 0.4) 
    --[[local displayed_img = TextDataset.Compose(tmp_im, {TextDataset.RandomSizedCrop(tmp_im, 224), 
            TextDataset.Lighting(tmp_im, 0.1, pca.eigval, pca.eigvec),
            TextDataset.ColorNormalize(tmp_im, meanstd),
            TextDataset.HorizontalFlip(tmp_im, 0.5),
            })]]--
         

   -- display.image({tmp_im, displayed_img}, {win='final'})
    image.save(filename_left, color_jitter_img)
end

return M.TextDataset 
