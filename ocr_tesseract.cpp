

#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"

#include <iostream>
#include <fstream>
#include <queue>

namespace cv
{
namespace text
{

using namespace std;

void OCRTesseract::run(Mat& image, string& output_text, vector<Rect>* component_rects,
                       vector<string>* component_texts, vector<float>* component_confidences,
                       int component_level)
{
    CV_Assert( (image.type() == CV_8UC1) || (image.type() == CV_8UC3) );
    CV_Assert( (component_level == OCR_LEVEL_TEXTLINE) || (component_level == OCR_LEVEL_WORD) );
    output_text.clear();
    if (component_rects != NULL)
        component_rects->clear();
    if (component_texts != NULL)
        component_texts->clear();
    if (component_confidences != NULL)
        component_confidences->clear();
}

class OCRTesseractImpl : public OCRTesseract
{
private:
#ifdef HAVE_TESSERACT
    tesseract::TessBaseAPI tess;
#endif

public:
    //Default constructor
    OCRTesseractImpl(const char* datapath, const char* language, const char* char_whitelist, int oemode, int psmode)
    {

#ifdef HAVE_TESSERACT
        const char *lang = "eng";
        if (language != NULL)
            lang = language;
		datapath = "F:\\Program Files\\Tesseract-OCR";

        if (tess.Init(datapath, lang, (tesseract::OcrEngineMode)oemode))
        {
            cout << "OCRTesseract: Could not initialize tesseract." << endl;
            throw 1;
        }

        //cout << "OCRTesseract: tesseract version " << tess.Version() << endl;

        tesseract::PageSegMode pagesegmode = (tesseract::PageSegMode)psmode;
        tess.SetPageSegMode(pagesegmode);

        if(char_whitelist != NULL)
            tess.SetVariable("tessedit_char_whitelist", char_whitelist);
        else
            tess.SetVariable("tessedit_char_whitelist", "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");

        tess.SetVariable("save_best_choices", "T");
#else
        cout << "OCRTesseract("<<oemode<<psmode<<"): Tesseract not found." << endl;
        if (datapath != NULL)
            cout << "            " << datapath << endl;
        if (language != NULL)
            cout << "            " << language << endl;
        if (char_whitelist != NULL)
            cout << "            " << char_whitelist << endl;
#endif
    }

    ~OCRTesseractImpl()
    {
#ifdef HAVE_TESSERACT
        tess.End();
#endif
    }

    void run(Mat& image, string& output, vector<Rect>* component_rects=NULL,
             vector<string>* component_texts=NULL, vector<float>* component_confidences=NULL,
             int component_level=0)
    {

        CV_Assert( (image.type() == CV_8UC1) || (image.type() == CV_8UC1) );

#ifdef HAVE_TESSERACT

        if (component_texts != 0)
            component_texts->clear();
        if (component_rects != 0)
            component_rects->clear();
        if (component_confidences != 0)
            component_confidences->clear();

        tess.SetImage((uchar*)image.data, image.size().width, image.size().height, image.channels(), image.step1());
        tess.Recognize(0);
        output = string(tess.GetUTF8Text());

        if ( (component_rects != NULL) || (component_texts != NULL) || (component_confidences != NULL) )
        {
            tesseract::ResultIterator* ri = tess.GetIterator();
            tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
            if (component_level == OCR_LEVEL_TEXTLINE)
                level = tesseract::RIL_TEXTLINE;

            if (ri != 0) {
                do {
                    const char* word = ri->GetUTF8Text(level);
                    if (word == NULL)
                        continue;
                    float conf = ri->Confidence(level);
                    int x1, y1, x2, y2;
                    ri->BoundingBox(level, &x1, &y1, &x2, &y2);

                    if (component_texts != 0)
                        component_texts->push_back(string(word));
                    if (component_rects != 0)
                        component_rects->push_back(Rect(x1,y1,x2-x1,y2-y1));
                    if (component_confidences != 0)
                        component_confidences->push_back(conf);

                  //  delete[] word;
                } while (ri->Next(level));
            }
            delete ri;
        }

        tess.Clear();

#else

        cout << "OCRTesseract(" << component_level << image.type() <<"): Tesseract not found." << endl;
        output.clear();
        if(component_rects)
            component_rects->clear();
        if(component_texts)
            component_texts->clear();
        if(component_confidences)
            component_confidences->clear();
#endif
    }


};

Ptr<OCRTesseract> OCRTesseract::create(const char* datapath, const char* language,
                                       const char* char_whitelist, int oem, int psmode)
{
    return makePtr<OCRTesseractImpl>(datapath,language,char_whitelist,oem,psmode);
}


}
}
