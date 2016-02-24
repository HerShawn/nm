

#ifndef __OPENCV_TEXT_OCR_HPP__
#define __OPENCV_TEXT_OCR_HPP__

#include <vector>
#include <string>

namespace cv
{
namespace text
{

//! @addtogroup text_recognize
//! @{

enum
{
    OCR_LEVEL_WORD,
    OCR_LEVEL_TEXTLINE
};

//base class BaseOCR declares a common API that would be used in a typical text recognition scenario
class CV_EXPORTS BaseOCR
{
public:
    virtual ~BaseOCR() {};
    virtual void run(Mat& image, std::string& output_text, std::vector<Rect>* component_rects=NULL,
                     std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL,
                     int component_level=0) = 0;
};

/** @brief OCRTesseract class provides an interface with the tesseract-ocr API (v3.02.02) in C++.

Notice that it is compiled only when tesseract-ocr is correctly installed.

@note
   -   (C++) An example of OCRTesseract recognition combined with scene text detection can be found
        at the end_to_end_recognition demo:
        <https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/end_to_end_recognition.cpp>
    -   (C++) Another example of OCRTesseract recognition combined with scene text detection can be
        found at the webcam_demo:
        <https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/webcam_demo.cpp>
 */
class CV_EXPORTS OCRTesseract : public BaseOCR
{
public:
    /** @brief Recognize text using the tesseract-ocr API.

    Takes image on input and returns recognized text in the output_text parameter. Optionally
    provides also the Rects for individual text elements found (e.g. words), and the list of those
    text elements with their confidence values.

    @param image Input image CV_8UC1 or CV_8UC3
    @param output_text Output text of the tesseract-ocr.
    @param component_rects If provided the method will output a list of Rects for the individual
    text elements found (e.g. words or text lines).
    @param component_texts If provided the method will output a list of text strings for the
    recognition of individual text elements found (e.g. words or text lines).
    @param component_confidences If provided the method will output a list of confidence values
    for the recognition of individual text elements found (e.g. words or text lines).
    @param component_level OCR_LEVEL_WORD (by default), or OCR_LEVEL_TEXT_LINE.
     */
    virtual void run(Mat& image, std::string& output_text, std::vector<Rect>* component_rects=NULL,
                     std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL,
                     int component_level=0);

    /** @brief Creates an instance of the OCRTesseract class. Initializes Tesseract.

    @param datapath the name of the parent directory of tessdata ended with "/", or NULL to use the
    system's default directory.
    @param language an ISO 639-3 code or NULL will default to "eng".
    @param char_whitelist specifies the list of characters used for recognition. NULL defaults to
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".
    @param oem tesseract-ocr offers different OCR Engine Modes (OEM), by deffault
    tesseract::OEM_DEFAULT is used. See the tesseract-ocr API documentation for other possible
    values.
    @param psmode tesseract-ocr offers different Page Segmentation Modes (PSM) tesseract::PSM_AUTO
    (fully automatic layout analysis) is used. See the tesseract-ocr API documentation for other
    possible values.
     */
    static Ptr<OCRTesseract> create(const char* datapath=NULL, const char* language=NULL,
                                    const char* char_whitelist=NULL, int oem=3, int psmode=3);
};


/* OCR HMM Decoder */

enum decoder_mode
{
    OCR_DECODER_VITERBI = 0 // Other algorithms may be added
};

/** @brief OCRHMMDecoder class provides an interface for OCR using Hidden Markov Models.

@note
   -   (C++) An example on using OCRHMMDecoder recognition combined with scene text detection can
        be found at the webcam_demo sample:
        <https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/webcam_demo.cpp>
 */
class CV_EXPORTS OCRHMMDecoder : public BaseOCR
{
public:

    /** @brief Callback with the character classifier is made a class.

    This way it hides the feature extractor and the classifier itself, so developers can write
    their own OCR code.

    The default character classifier and feature extractor can be loaded using the utility funtion
    loadOCRHMMClassifierNM and KNN model provided in
    <https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/OCRHMM_knn_model_data.xml.gz>.
     */
    class CV_EXPORTS ClassifierCallback
    {
    public:
        virtual ~ClassifierCallback() { }
        /** @brief The character classifier must return a (ranked list of) class(es) id('s)

        @param image Input image CV_8UC1 or CV_8UC3 with a single letter.
        @param out_class The classifier returns the character class categorical label, or list of
        class labels, to which the input image corresponds.
        @param out_confidence The classifier returns the probability of the input image
        corresponding to each classes in out_class.
         */
        virtual void eval( InputArray image, std::vector<int>& out_class, std::vector<double>& out_confidence);
    };

public:
    /** @brief Recognize text using HMM.

    Takes image on input and returns recognized text in the output_text parameter. Optionally
    provides also the Rects for individual text elements found (e.g. words), and the list of those
    text elements with their confidence values.

    @param image Input image CV_8UC1 with a single text line (or word).

    @param output_text Output text. Most likely character sequence found by the HMM decoder.

    @param component_rects If provided the method will output a list of Rects for the individual
    text elements found (e.g. words).

    @param component_texts If provided the method will output a list of text strings for the
    recognition of individual text elements found (e.g. words).

    @param component_confidences If provided the method will output a list of confidence values
    for the recognition of individual text elements found (e.g. words).

    @param component_level Only OCR_LEVEL_WORD is supported.
     */
    virtual void run(Mat& image, std::string& output_text, std::vector<Rect>* component_rects=NULL,
                     std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL,
                     int component_level=0);

    /** @brief Creates an instance of the OCRHMMDecoder class. Initializes HMMDecoder.

    @param classifier The character classifier with built in feature extractor.

    @param vocabulary The language vocabulary (chars when ascii english text). vocabulary.size()
    must be equal to the number of classes of the classifier.

    @param transition_probabilities_table Table with transition probabilities between character
    pairs. cols == rows == vocabulary.size().

    @param emission_probabilities_table Table with observation emission probabilities. cols ==
    rows == vocabulary.size().

    @param mode HMM Decoding algorithm. Only OCR_DECODER_VITERBI is available for the moment
    (<http://en.wikipedia.org/wiki/Viterbi_algorithm>).
     */
    static Ptr<OCRHMMDecoder> create(const Ptr<OCRHMMDecoder::ClassifierCallback> classifier,// The character classifier with built in feature extractor
                                     const std::string& vocabulary,                    // The language vocabulary (chars when ascii english text)
                                                                                       //     size() must be equal to the number of classes
                                     InputArray transition_probabilities_table,        // Table with transition probabilities between character pairs
                                                                                       //     cols == rows == vocabulari.size()
                                     InputArray emission_probabilities_table,          // Table with observation emission probabilities
                                                                                       //     cols == rows == vocabulari.size()
                                     decoder_mode mode = OCR_DECODER_VITERBI);         // HMM Decoding algorithm (only Viterbi for the moment)

protected:

    Ptr<OCRHMMDecoder::ClassifierCallback> classifier;
    std::string vocabulary;
    Mat transition_p;
    Mat emission_p;
    decoder_mode mode;
};

/** @brief Allow to implicitly load the default character classifier when creating an OCRHMMDecoder object.

@param filename The XML or YAML file with the classifier model (e.g. OCRHMM_knn_model_data.xml)

The default classifier is based in the scene text recognition method proposed by Lukás Neumann &
Jiri Matas in [Neumann11b]. Basically, the region (contour) in the input image is normalized to a
fixed size, while retaining the centroid and aspect ratio, in order to extract a feature vector
based on gradient orientations along the chain-code of its perimeter. Then, the region is classified
using a KNN model trained with synthetic data of rendered characters with different standard font
types.
 */
CV_EXPORTS Ptr<OCRHMMDecoder::ClassifierCallback> loadOCRHMMClassifierNM(const std::string& filename);

//! @}

}
}
#endif // _OPENCV_TEXT_OCR_HPP_
