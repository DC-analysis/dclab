def get_inertiaratio(cont,pos_x=[],pos_y=[],align=False):
    """
    Calculate the inertia ratio (using second moment of inertia or principal
    second moment of inertia if align=True)
    
    Parameters
    ----------
    cont: ndarray or list of ndarrays of shape (N,2)
        A 2D array that holds the contour of an event (in pixels)
        e.g. obtained using `mm.contour` where  `mm` is an instance
        of `RTDCBase`. The first and second columns of `cont`
        correspond to the x- and y-coordinates of the contour.
    pos_x: float or ndarray 
        The x coordinate of the centroid of the event(s)
        e.g. obtained using mm.pos_x  (mm is an instance of RTDC_DataSet)
    pos_y: float or ndarray 
        The y coordinate of the centroid of the event(s)
        e.g. obtained using mm.pos_lat  (mm is an instance of RTDC_DataSet)
    align: True or False, wether the contour should be re-oriented such that the
        longest axis meets the x-axis (Principal moment of ineria)
    Returns
    -------
    inertiaratio_raw: float or ndarray
        inertia ratio of the contour
    inertiaratio_hull: float or ndarray
        inertia ratio of the convex hull of the contour
    inertiaratio_principal_raw: float or ndarray
        principal inertia ratio of the contour
    inertiaratio_principal_hull: float or ndarray
        principal inertia ratio of the convex hull of the contour
    orientation: float or ndarray
        orientation of the object (with respect to x/y axis) in radians
    Unit = 1    
    """
    if not type(cont)==list:
        cont = [cont]
    
    #Some checks
    #is pos_x and pos_y provided?
    if len(pos_x) != 0: #if pos_x is not an empty array
        get_pos_x = False
        msg = 'Unequal nr. of given contours and pos_x - values'
        assert len(pos_x) == len(cont),msg
    else: #if not provided it will be calculated later
        get_pos_x = True
    if len(pos_y) != 0: #if pos_x is not an empty array
        get_pos_y = False 
        msg = 'Unequal nr. of given contours and pos_y - values'
        assert len(pos_y) == len(cont),msg
    else: #if not provided it will be calculated later
        get_pos_y = True 
        
        
            
    # results are stored in a separate array initialized with nans
    inertiaratio_raw = np.zeros(len(cont), dtype=float)*np.nan
    inertiaratio_hull = np.zeros(len(cont), dtype=float)*np.nan
                                
    if align:
        # results are stored in a separate array initialized with nans
        inertiaratio_principal_raw = np.zeros(len(cont), dtype=float)*np.nan
        inertiaratio_principal_hull = np.zeros(len(cont), dtype=float)*np.nan
        orientations = np.zeros(len(cont), dtype=float)*np.nan
    
    #convert contours to OpenCV format
    cont = [a.astype(int).reshape(a.shape[0],1,2) for a in cont] 
    
    for i in range(len(cont)):
        hull = cv2.convexHull(cont[i], returnPoints=True)
        mu1 = cv2.moments(cont[i], False)
        mu2 = cv2.moments(hull, False)
    
        inertiaratio_raw[i] = np.sqrt(mu1['mu20']/mu1['mu02'])
        inertiaratio_hull[i] = np.sqrt(mu2['mu20']/mu2['mu02'])

        if align:                                 
            #get the orientation of the contour
            if mu1['mu02'] == mu1['mu20']:
                orientation = 0.5 * np.pi
            else:
                orientation = 0.5 * np.arctan(2 * mu1['mu11']/(mu1['mu02'] - mu1['mu20']))
            I_1 = abs(0.5 * (mu1['mu02'] + mu1['mu20']) + 0.5*(mu1['mu02'] - mu1['mu20']) * np.cos(2*orientation) + mu1['mu11'] * np.sin(2*orientation))  # x-axis tilted by alpha
            I_2 = abs(0.5 * (mu1['mu02'] + mu1['mu20']) - 0.5*(mu1['mu02'] - mu1['mu20']) * np.cos(2*orientation) - mu1['mu11'] * np.sin(2*orientation))  # y-axis tilted by alpha
            
            # If I_1 is bigger than I_2, the angle needs to be corrected by pi/2  
            # convention for arctan function is to take value between -pi/2 and pi/2
            if I_1 > I_2:
                orientation = (orientation + 0.5 * np.pi)     
            orientations[i] = orientation
            #Take the centroid if given or calculate if not given
            if get_pos_x:
                #calculate pos_x
                pos_x_i = mu1['m10']/mu1['m00']
            else:
                pos_x_i = pos_x[i]
                
            if get_pos_y:
                #calculate pos_y
                pos_y_i = mu1['m01']/mu1['m00']
            else:
                pos_y_i = pos_y[i]

            #remove the centroid (move the cell such that its centroid equla the origin)
            cont_i = cont[i].astype(float).reshape(cont[i].shape[0],2)
            cont_i[:,0] = cont_i[:,0]-pos_x_i
            cont_i[:,1] = cont_i[:,1]-pos_y_i
            #Rotate the contour
            rho = np.sqrt(cont_i[:,0]**2 + cont_i[:,1]**2) #get polarcoordinates
            phi = np.arctan2(cont_i[:,1], cont_i[:,0])
            phi = phi+orientation #rotate
            #transform back to cartesian coordinates
            cont_i[:,0] = rho * np.cos(phi)
            cont_i[:,1] = rho * np.sin(phi)

            #convert contour to OpenCV format
            cont_i = cont_i.astype(int).reshape(cont_i.shape[0],1,2)
    
            #get the moments of this rotated object
            hull = cv2.convexHull(cont_i, returnPoints=True)
            mu1 = cv2.moments(cont_i, False)
            mu2 = cv2.moments(hull, False)
        
            inertiaratio_principal_raw[i] = np.sqrt(mu1['mu20']/mu1['mu02'])
            inertiaratio_principal_hull[i] = np.sqrt(mu2['mu20']/mu2['mu02'])

    if not type(cont)==list:
        # Do not return a list if the input contour was not a list
        inertiaratio_raw = inertiaratio_raw[0]
        inertiaratio_hull = inertiaratio_hull[0]
        inertiaratio_principal_raw = inertiaratio_principal_raw[0]
        inertiaratio_principal_hull = inertiaratio_principal_hull[0]
        orientations = orientations[0]

    if align:
        return(inertiaratio_raw,inertiaratio_hull,inertiaratio_principal_raw,inertiaratio_principal_hull,orientations)
    else:
        return(inertiaratio_raw,inertiaratio_hull)
