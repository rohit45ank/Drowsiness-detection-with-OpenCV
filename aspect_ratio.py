from scipy.spatial import distance

def aspect_ratio(eye):
    disa = distance.euclidean(eye[1], eye[5])
    disb = distance.euclidean(eye[2], eye[4])
    disc = distance.euclidean(eye[0], eye[3])

    ratio = (disa+disb) / (2*disc)
    return ratio

    #returns the aspect ratio for the given eye
