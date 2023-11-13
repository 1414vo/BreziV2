import numpy as np

def surrounding_coords(coords):
    return [(coords[0] - 1, coords[1] - 1),(coords[0] - 1, coords[1] + 0),(coords[0] - 1, coords[1] + 1),
            (coords[0] + 0, coords[1] - 1),                               (coords[0] + 0, coords[1] + 1),
            (coords[0] + 1, coords[1] - 1),(coords[0] + 1, coords[1] + 0),(coords[0] + 1, coords[1] + 1)]

def get_image_groups(mask):
    queue = []
    traversed = np.zeros(mask.shape, dtype = bool)
    groups = []
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if not traversed[i][j]:
                # Queue of region pixels
                group = [(i, j)]
                queue = [(i, j)]
                while len(queue) > 0:
                    coords = queue.pop()
                    # Iterate over surrounding pixels
                    for x, y in surrounding_coords(coords):
                        if 0 <= x < len(mask) and  0 <= y < len(mask[0]):
                            if mask[x][y] and not traversed[x][y]:
                                queue.append((x, y))
                                group.append((x,y))
                                traversed[x, y] = True
                groups.append(mask[np.array(group).transpose()])
    return groups

def register_images(img, ref):
        
    # Convert to grayscale. 
    img_gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ref_gr = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) 
    height, width = ref_gr.shape 
    
    # Create ORB detector with 5000 features. 
    orb_detector = cv2.ORB_create(5000) 
    
    # Find keypoints and descriptors. 
    # The first arg is the image, second arg is the mask 
    #  (which is not required in this case). 
    kp1, d1 = orb_detector.detectAndCompute(img_gr, None) 
    kp2, d2 = orb_detector.detectAndCompute(ref_gr, None) 
    
    # Match features between the two images. 
    # We create a Brute Force matcher with  
    # Hamming distance as measurement mode. 
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
    
    # Match the two sets of descriptors. 
    matches = list(matcher.match(d1, d2))
    
    # Sort matches on the basis of their Hamming distance. 
    matches.sort(key = lambda x: x.distance) 
    
    # Take the top 90 % matches forward. 
    matches = matches[:int(len(matches)*0.9)] 
    no_of_matches = len(matches) 
    
    # Define empty matrices of shape no_of_matches * 2. 
    p1 = np.zeros((no_of_matches, 2)) 
    p2 = np.zeros((no_of_matches, 2)) 
    
    for i in range(len(matches)): 
        p1[i, :] = kp1[matches[i].queryIdx].pt 
        p2[i, :] = kp2[matches[i].trainIdx].pt 
    
    # Find the homography matrix. 
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 
    
    # Use this matrix to transform the 
    # colored image wrt the reference image. 
    transformed_img = cv2.warpPerspective(img, 
                        homography, (width, height)) 
    