def WatershedSegmentation (img, n_seeds, alpha):
    get_ipython().run_line_magic('matplotlib', '')
    img_original = cv2.imread(img)
    plt.imshow((img_original))
    #particular command we are going to use to select multiple seeds on the image (multiple points)
    coordinates = plt.ginput(n_seeds)
    #to obtain certain coordinates selected on the image:
    x = []
    y = []
    
    for i in coordinates:
        #Round float numbers to nearest integers to obtain coordinates
        x_coord = int(round(i[1])) 
        y_coord = int(round(i[0]))
        #Append the values to x and y lists that have been already created
        x.append(x_coord)
        y.append(y_coord)
        
    get_ipython().run_line_magic('matplotlib', 'inline')
    print('')
    #Print the coordinates of the points that have been selecteed on the image
    print('The gray values associated to the seeds are:', img_original[x,y])
    
    #printing the coordinates of the multiple seeds selected
    coords = []
    for xcoord, ycoord in zip (x,y):
        coords.append([xcoord, ycoord])
        print('Coordinates of the seed', coords.index([xcoord, ycoord]) +1 ,
              'selected on the image:', [xcoord, ycoord])
    print('')

    #SEGMENTATION MASK 1: WITHOUT USING IMIMPOSEMIN FUNCTION:
    
    #1. Calculation of the gradient image,being obtained by Sobel derivative filter:
    img = io.imread(img, as_gray=True)
    img = np.array(img)
    
    #plotting the original image
    plt.figure(figsize=(10,5))
    plt.title('Original image', color = 'skyblue', size=15)
    plt.imshow(img, cmap = plt.cm.gray)
    #applying the Sobel filter to the image 
    sobel_filt = filters.sobel(img)
    sobel_filt *= 255 / (np.max(sobel_filt))
    #plotting the gradient image
    plt.figure(figsize=(10,5))
    plt.title('Sobel filter', color = 'skyblue', size=15)
    plt.imshow(sobel_filt, cmap = plt.cm.gray, alpha = alpha)
    #saving the gradient image in your pc
    cv2.imwrite('Sobel_filt.png',sobel_filt)
    
    #2. Computation of the binary mask, obtained after filtering hthe input image:
    return_frame, bin_mask = cv2.threshold(sobel_filt, 30,150, cv2.THRESH_BINARY)
    plt.figure(figsize=(10,5))
    plt.title('Binary mask of gradient img', color = 'skyblue', size=15)
    plt.imshow(bin_mask, cmap = plt.cm.gray)
    
    #3.Inverse of the euclidean distance transform applied to the binary mask
    #generation of a (euclidean) distance transformed image 
    euclidean_dist = ndi.distance_transform_edt(bin_mask)
    euclidean_dist = np.uint8(euclidean_dist)
    plt.figure(figsize=(10,5))
    plt.title('Distance transformation', color = 'skyblue', size=15)
    plt.imshow(euclidean_dist, cmap = plt.cm.gray)
    
    #4.Watershed segmentation without imimposemin and with the binary mask + distance transformed image
    local_max = peak_local_max (euclidean_dist, indices = False, labels = bin_mask)
    markers = ndi.label(local_max)[0] #labeling the peaks in an array
    #watershed transformation:
    WS = watershed(-euclidean_dist, markers, watershed_line = True)
    plt.figure(figsize=(10,5))
    plt.title('Segmentation without using imimposemin (colormap)', color = 'skyblue', size=15)
    plt.imshow(WS, cmap = plt.cm.nipy_spectral)
    plt.figure(figsize=(10,5))
    plt.title('Segmentation without using imimposemin (graymap)', color = 'skyblue', size=15)
    plt.imshow(WS, cmap = plt.cm.gray)
    #SEGMENTATION MASK 2: USING IMIMPOSEMIN FUNCTION
    
    #1. Creation of the binary mask
    #let's create the zero matrix 
    
    height, width = img.shape[0], img.shape[1]
    print('Dimensions of the image we have:',height,'x' ,width)
    zeros_matrix = np.zeros((height, width)) #matrix of zeros of the same size as the image
    print('')
    print('X coordinates of the nº of seeds selected:',x)
    print('X coordinates of the nº of seeds selected:',y)
    
    for i in range(0, len(x)): #converting the specific coordinates of each seed, already selected
        #on the image, into a 1 value in our zero matrix 
        zeros_matrix[x[i],y[i]] =255
        cv2.imwrite('binmask.png',zeros_matrix)
    
    #and finally, let's make use of the imimposemin function that we have available in aula virtual
    #imimposemin(sobel_filt,zeros_matrix)
    sgm2 = imimposemin(sobel_filt,zeros_matrix)
    plt.figure(figsize=(10,5))
    plt.title('Segmentation using imimposemin(colormap)', color = 'skyblue', size=15)
    plt.imshow(sgm2, cmap = plt.cm.nipy_spectral, alpha = alpha)
    plt.figure(figsize=(10,5))
    plt.title('Segmentation using imimposemin(graymap)', color = 'skyblue', size=15)
    plt.imshow(sgm2, cmap = plt.cm.gray, alpha = alpha)
    
    return 
