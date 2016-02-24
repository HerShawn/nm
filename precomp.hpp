

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#include "text.hpp"

#include "text_config.hpp"

#ifdef HAVE_TESSERACT
#include <baseapi.h>
#include <resultiterator.h>
#pragma  comment(lib,"libtesseract302d.lib")
#endif

#endif
