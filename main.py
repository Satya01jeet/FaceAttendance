import face_recognition as fr
import cv2
import numpy as np
import os
# making a path to folder where known faces are kept
path = 'knownfaces'
knownFaceList = os.listdir(path)

# initializing empty classes to store image and names
imageClass = []
nameClass = []

# print(knownFaceList)

# looping through known faces and storing
for face in knownFaceList:
    currFace = cv2.imread(f'{path}/{face}')
    imageClass.append(currFace)
    nameClass.append(os.path.splitext(face)[0])

print(imageClass)
# print(nameClass)

# Fundtion to get face encodings of all the known faces (to match them with target faces later on),
def knownFaceEncodings(imageClass):
    encodingsList = []
    for fimg in imageClass:
        fimg = cv2.cvtColor(fimg, cv2.COLOR_BGR2RGB)
        encd = fr.face_encodings(fimg)[0]
        encodingsList.append(encd)
    return encodingsList


knownFaces = knownFaceEncodings(imageClass)
print(knownFaces)

# Function to update detected faces in the csv file.
def markAttendence(name):
    with open('Attendence.csv', 'r+') as f:
        DataList = f.readlines()
        nList = []
        for line in DataList:
            entry = line.split(',')
            nList.append(entry[0])
        if name not in nList:
            f.writelines(f'\n{name}')



print("encodings complete.")

# Reading image and converting it to RBG format.
image = cv2.imread('sample.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Getting the locations and encodings of faces in the target image.
currFaceLoc = fr.face_locations(image)
currFaceencodings = fr.face_encodings(image, currFaceLoc)

for encodeFace, facesLoc in zip(currFaceencodings, currFaceLoc):
    '''
    takes both face locations and face encodings of each location from facesInCurrFrame and encodingsInCurrFrame 
    respectively.
    '''
    matches = fr.compare_faces(knownFaces, encodeFace)  # comparing faces in known list with curr image
    faceDis = fr.face_distance(knownFaces, encodeFace)  # finding distance
    # print(faceDis)
    matchIndex = np.argmin(faceDis)

    # if faces in target image matches with faces in knownFaces folder
    if matches[matchIndex]:
        name = nameClass[matchIndex].upper()
        # print(name)
        y1, x2, y2, x1 = facesLoc
        y1, x2, y2, x1 = y1, x2, y2, x1
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        markAttendence(name)

cv2.imshow("Group of people", image)
cv2.waitKey(0)


print('end')
