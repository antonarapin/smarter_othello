import cImage, math

images = []
images.append(cImage.FileImage("blank.gif"))
images.append(cImage.FileImage("p1disc.gif"))
images.append(cImage.FileImage("p2disc.gif"))

for img in images:
    newImg = cImage.EmptyImage(math.floor(img.getWidth()*0.9),math.floor(img.getHeight()*0.9))
    
    yCount=0
    for y in range(img.getHeight()):
        if y%10!=0:
            xCount = 0
            for x in range(img.getWidth()):
                if x%10!=0:
                    newImg.setPixel(xCount,yCount,img.getPixel(x,y))
                    xCount+=1
            yCount+=1
    newImg.save(img.imFileName+".gif")
