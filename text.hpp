

#ifndef __OPENCV_TEXT_HPP__
#define __OPENCV_TEXT_HPP__

#include "erfilter.hpp"
#include "ocr.hpp"

/** @defgroup text Scene Text Detection and Recognition

The opencv_text module provides different algorithms for text detection and recognition in natural
scene images.

  @{
    @defgroup text_detect Scene Text Detection

Class-specific Extremal Regions for Scene Text Detection
--------------------------------------------------------

The scene text detection algorithm described below has been initially proposed by Lukás Neumann &
Jiri Matas [Neumann12]. The main idea behind Class-specific Extremal Regions is similar to the MSER
in that suitable Extremal Regions (ERs) are selected from the whole component tree of the image.
However, this technique differs from MSER in that selection of suitable ERs is done by a sequential
classifier trained for character detection, i.e. dropping the stability requirement of MSERs and
selecting class-specific (not necessarily stable) regions.

The component tree of an image is constructed by thresholding by an increasing value step-by-step
from 0 to 255 and then linking the obtained connected components from successive levels in a
hierarchy by their inclusion relation:

![image](pics/component_tree.png)

The component tree may conatain a huge number of regions even for a very simple image as shown in
the previous image. This number can easily reach the order of 1 x 10\^6 regions for an average 1
Megapixel image. In order to efficiently select suitable regions among all the ERs the algorithm
make use of a sequential classifier with two differentiated stages.

In the first stage incrementally computable descriptors (area, perimeter, bounding box, and euler
number) are computed (in O(1)) for each region r and used as features for a classifier which
estimates the class-conditional probability p(r|character). Only the ERs which correspond to local
maximum of the probability p(r|character) are selected (if their probability is above a global limit
p_min and the difference between local maximum and local minimum is greater than a delta_min
value).

In the second stage, the ERs that passed the first stage are classified into character and
non-character classes using more informative but also more computationally expensive features. (Hole
area ratio, convex hull ratio, and the number of outer boundary inflexion points).

This ER filtering process is done in different single-channel projections of the input image in
order to increase the character localization recall.

After the ER filtering is done on each input channel, character candidates must be grouped in
high-level text blocks (i.e. words, text lines, paragraphs, ...). The opencv_text module implements
two different grouping algorithms: the Exhaustive Search algorithm proposed in [Neumann11] for
grouping horizontally aligned text, and the method proposed by Lluis Gomez and Dimosthenis Karatzas
in [Gomez13][Gomez14] for grouping arbitrary oriented text (see erGrouping).

To see the text detector at work, have a look at the textdetection demo:
<https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/textdetection.cpp>

    @defgroup text_recognize Scene Text Recognition
  @}
*/

#endif
