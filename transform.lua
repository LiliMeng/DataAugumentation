-- Image transforms for data augmentation and input normalization
-- Lili@Xtract May 16th


image = require 'image'
display = require 'display'


    local im = image.load('../../../../PDFreceipts/Scanned_Reciepts/Image_00185.jpg',3,'byte')
    local tmp_im=im:clone()
    local height,width  = im:size(2), im:size(3)
    print(height)
    print(width)
    
    for i=1, 5 do
    	local theta = 2*i
   		local uim_left=image.rotate(tmp_im, theta*math.pi/180,'bilinear')
   		local uim_right=image.rotate(tmp_im, -theta*math.pi/180,'bilinear')
   		local uim_stretch=image.scale(tmp_im, 1.1*i*width, height,'bicubic')
  
	    display.image({im, uim}, {win='final'})
	
	    local filename_left="Image_00185_l_"..i..".jpg"
	    local filename_right="Image_00185_r_"..i..".jpg"
	    local filename_stretch="Image_00185_s_"..i..".jpg"
	    image.save(filename_left,uim_left)
	    image.save(filename_right,uim_right)
	    image.save(filename_stretch,uim_stretch)
	end
    
