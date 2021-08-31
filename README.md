# Facemorphing
Morphes two faces into one

## Dependencies:
[Download the trained Model for facial landmark recognition](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)

### Results:
James Bond + Scarlet Johansson:
![Result](results/result.png?raw=true "-")

### Alphablending between two faces:
![Result](results/gif.gif?raw=true "-")

### Morph matrix:
![Result](results/matrix4x7.png?raw=true "-")

### How it works:
Find the landmarks in both images (left and right):
![Landmarks](results/landmarks.png?raw=true "-")

Use the [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation) to generate individual areas (triangles) from the landmarks:
![Delnuay](results/delunay.png?raw=true "-")

Merge corresponding triangles together.
![Howto](results/howto.JPG?raw=true "-")
