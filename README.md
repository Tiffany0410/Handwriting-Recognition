# Handwriting-Recognition

This project is center around implementing solutions to image recognition problems, which is the basic building block of many applications, ranging from photo apps like Google Lens or Face recognition to self-driving cars.
The specific problem that this project will focus on is handwriting recognition. More specifically, this program can read in images of hand-written digits and determine which digit the image shows.


## Image Format
The PGM (or Portable Gray Map) image format allows for images that are encoded in human-readable ASCII text (as opposed to a binary format), so the contents can be viewed in a text editor. You can think of an image as a matrix of pixels. Each pixel for a grayscale image is encoded by one integer number representing the grayscale of the pixel. A PGM image file contains information on all the pixels in the image (in the body of the image file), preceded by a header. The 31 lines below are a sample pgm file representing the digit '9'. 

## Algorithms
- Parallel K-Nearest-Neighbour (kNN)
  -  parallelizing the code to run over multiple processes which can be distributed over the the cores in the processor. 
- Decision Tree
