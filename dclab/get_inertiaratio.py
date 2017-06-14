def get_inertiaratio(cont):
    """
    Calculate the inertia ratio
    
    Parameters
    ----------
    cont: ndarray or list of ndarrays of shape (N,2)
        A 2D array that holds the contour of an event (in pixels)
        e.g. obtained using `mm.contour` where  `mm` is an instance
        of `RTDCBase`. The first and second columns of `cont`
        correspond to the x- and y-coordinates of the contour.

    Returns
    -------
    inertiaratio_raw: float or ndarray
        inertia ratio of the contour
    inertiaratio_hull: float or ndarray
        inertia ratio of the convex hull of the contour
    Unit = 1    
    """
    if not type(cont)==list:
        cont = [cont]
        
    # results are stored in a separate array initialized with nans
    inertiaratio_raw = np.zeros(len(cont), dtype=float)*np.nan
    inertiaratio_hull = np.zeros(len(cont), dtype=float)*np.nan
    
    #convert contours to OpenCV format
    cont = [a.astype(int).reshape(a.shape[0],1,2) for a in cont] 
    
    for i in range(len(cont)):
        hull = cv2.convexHull(cont[i], returnPoints=True)
        mu1 = cv2.moments(cont[i], False)
        mu2 = cv2.moments(hull, False)
    
        inertiaratio_raw[i] = np.sqrt(mu1['mu20']/mu1['mu02'])
        inertiaratio_hull[i] = np.sqrt(mu2['mu20']/mu2['mu02'])

    if not type(cont)==list:
        # Do not return a list if the input contour was not a list
        inertiaratio_raw = inertiaratio_raw[0]
        inertiaratio_hull = inertiaratio_hull[0]
        
    return(inertiaratio_raw,inertiaratio_hull)
