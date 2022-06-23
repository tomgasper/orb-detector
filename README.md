# ORB detector

Naive implementation of Oriented FAST and rotated BRIEF (ORB) feature detector.

>Oriented FAST and rotated BRIEF (ORB) is a fast robust local feature detector that can be used in computer vision tasks like object recognition or 3D reconstruction. It is based on the FAST keypoint detector and a modified version of the visual descriptor BRIEF (Binary Robust Independent Elementary Features). Its aim is to provide a fast and efficient alternative to SIFT.
>
>(source: https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF)

![Matches](https://github.com/tomgasper/orb-detector/blob/main/examples/match_0.jpg?raw=true)
![Rotated image matches](https://github.com/tomgasper/orb-detector/blob/main/examples/prev_1.jpg?raw=true)

# Notes
FAST implementation in this application is 10-15x slower than the SIMD version so to get way better performance you could just replace it with openCV implementation. Descriptor and matcher should have very similiar performance as openCV.

## Dependencies

* C++  
* OpenCV 3.4.14