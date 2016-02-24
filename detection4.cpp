
#include "text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <io.h>

using namespace std;
using namespace cv;
using namespace cv::text;
//在evaluation时所需的编辑距离的声明
//Calculate edit distance netween two words
size_t edit_distance(const string& A, const string& B);
size_t min(size_t x, size_t y, size_t z);
bool   isRepetitive(const string& s);
bool   sort_by_lenght(const string &a, const string &b);
//图割
//Draw ER's in an image via floodFill
void   er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation);
//导入一个目录下面的所有文件
void getFiles(string path, vector<string>& files)
{
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}


//Neuamnn端对端方法程序入口处：提取ER树+Neuamnn多通道、两阶段分类+剪枝穷举搜索文本行合并+利用编辑距离求查全、查准、WRA
//Perform text detection and recognition and evaluate results using edit distance
int main(int argc, char* argv[])
{
	argc = 3;

	float total_precision = 0;
	float total_recall = 0;
	float total_edit_distance_ratio = 0;
	float total_time_detection = 0;
	float total_time_grouping = 0;
	float total_time_ocr = 0;

	vector<string> files;
	char * filePath = "D:\\hx\\edgebox-contour-neumann\\Challenge2_Test_Task12_Images";
	////获取该路径下的所有文件  
	getFiles(filePath, files);

	int size = files.size();

	vector<string> file_names;

	for (int i = 0; i < size; i++)
	{
		string file_name = files[i].c_str();
		file_names.push_back(file_name);
	}

	/*vector<string> jpg_names;
	vector<string> txt_names;
	

	string flag = ".";
	size_t sub_pos1 = file_names[1].find_first_of(flag, 0) - 3;
	string ori_names = file_names[1].substr(0, sub_pos1);

	for (int i = 0; i < (size / 2); i++)
	{
		string flag = "jpg";
		size_t sub_pos1 = file_names[i].find(flag, 0) - 4;
		size_t sub_pos2 = 3;
		jpg_names.push_back(ori_names + file_names[i].substr(sub_pos1, sub_pos2) + ".jpg");

	}
*/
	/*for (int i = (size / 2); i < size; i++)
	{
		string flag = "gt_";
		size_t sub_pos1 = file_names[i].find(flag, 0);
		size_t sub_pos2 = 6;
		txt_names.push_back(ori_names + file_names[i].substr(sub_pos1, sub_pos2) + ".txt");
	}*/

	for (int i = 0; i < size; i++)
	{
		Mat image;
		if (argc > 1)
			image = imread(file_names[i]);
		else
		{
			return 0;
		}

		cout << endl << endl << "现在运行到第" << (i + 100) << "张图片" << endl;
		cout << "图片宽=" << image.cols << endl;
		cout << "图片高=" << image.rows << endl;

		vector<Mat> channels;
		Mat grey;
		cvtColor(image, grey, COLOR_RGB2GRAY);
		// Notice here we are only using grey channel, see textdetection.cpp for example with more channels
		channels.push_back(grey);
		channels.push_back(255 - grey);

		double t_d = (double)getTickCount();
		// Create ERFilter objects with the 1st and 2nd stage default classifiers
		Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"), 8, 0.00015f, 0.13f, 0.2f, true, 0.1f);
		Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"), 0.5);

		vector<vector<ERStat> > regions(channels.size());
		// Apply the default cascade classifier to each independent channel (could be done in parallel)
		for (int c = 0; c < (int)channels.size(); c++)
		{
			er_filter1->run(channels[c], regions[c]);
			er_filter2->run(channels[c], regions[c]);
		}

		double time_detection = ((double)getTickCount() - t_d) * 1000 / getTickFrequency();
		cout << "文字检测用时 = " << time_detection << endl;
		total_time_detection += time_detection;

		Mat out_img_decomposition = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
		vector<Vec2i> tmp_group;
		for (int i = 0; i < (int)regions.size(); i++)
		{
			for (int j = 0; j<(int)regions[i].size(); j++)
			{
				tmp_group.push_back(Vec2i(i, j));
			}
			Mat tmp = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
			er_draw(channels, regions, tmp_group, tmp);
			if (i > 0)
				tmp = tmp / 2;
			out_img_decomposition = out_img_decomposition | tmp;
			tmp_group.clear();
		}

		double t_g = (double)getTickCount();
		// Detect character groups
		vector< vector<Vec2i> > nm_region_groups;
		vector<Rect> nm_boxes;
		vector<Rect> temp_boxes;
		//七个通道的，就是在ergrouping这里导致了重复：H与255-S可能重合。重合的数据装在nm_region_groups和nm_boxes里面。
		erGrouping(image, channels, regions, nm_region_groups, nm_boxes, ERGROUPING_ORIENTATION_HORIZ);
		double time_grouping = ((double)getTickCount() - t_g) * 1000 / getTickFrequency();
		cout << "文本行合并用时 = " << time_grouping << endl;
		total_time_grouping += time_grouping;

		/*Text Recognition (OCR)*/
		double t_r = (double)getTickCount();
		Ptr<OCRTesseract> ocr = OCRTesseract::create();
		double time_ocr_initial = ((double)getTickCount() - t_r) * 1000 / getTickFrequency();
		cout << "启动OCR用时 = " << time_ocr_initial << endl;
		string output;
		Mat out_img;
		Mat out_img_detection;
		Mat out_img_segmentation = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
		image.copyTo(out_img);
		image.copyTo(out_img_detection);
		float scale_img = 600.f / image.rows;
		float scale_font = (float)(2 - scale_img) / 1.4f;
		vector<string> words_detection;
		t_r = (double)getTickCount();
		for (int i = 0; i < (int)nm_boxes.size(); i++)
		{
			//boundingbox就是由rectangle函数画出来的；这里的out_img_detection是文字检测的结果，
			//因此bounding box直接由文本行合并ergrouping得到的nm_boxes来画
			rectangle(out_img_detection, nm_boxes[i].tl(), nm_boxes[i].br(), Scalar(3, 128, 255), 2);

			Mat group_img = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);

			//vector< vector<Vec2i> >& nm_region_groups
			er_draw(channels, regions, nm_region_groups[i], group_img);
			Mat group_segmentation;
			group_img.copyTo(group_segmentation);
			//image(nm_boxes[i]).copyTo(group_img);
			group_img(nm_boxes[i]).copyTo(group_img);
			//copyMakeBorder复制图像并且制作边界。（处理边界卷积）
			copyMakeBorder(group_img, group_img, 15, 15, 15, 15, BORDER_CONSTANT, Scalar(0));

			vector<Rect>   boxes;
			vector<string> words;
			vector<float>  confidences;
			ocr->run(group_img, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);

			output.erase(remove(output.begin(), output.end(), '\n'), output.end());
			//cout << "OCR output = \"" << output << "\" lenght = " << output.size() << endl;
			if (output.size() < 3)
				continue;

			for (int j = 0; j < (int)boxes.size(); j++)
			{
				boxes[j].x += nm_boxes[i].x - 15;
				boxes[j].y += nm_boxes[i].y - 15;

				//cout << "  word = " << words[j] << "\t confidence = " << confidences[j] << endl;
				if ((words[j].size() < 2) || (confidences[j] < 51) ||
					((words[j].size() == 2) && (words[j][0] == words[j][1])) ||
					((words[j].size() < 4) && (confidences[j] < 60)) ||
					isRepetitive(words[j]))
					continue;
				cout << " Detected word = " << words[j] << "\t confidence = " << confidences[j] << endl;
				words_detection.push_back(words[j]);

				rectangle(out_img, boxes[j].tl(), boxes[j].br(), Scalar(255, 0, 255), 3);
				temp_boxes.push_back(boxes[j]);
				//Size word_size = getTextSize(words[j], FONT_HERSHEY_SIMPLEX, (double)scale_font, (int)(3 * scale_font), NULL);
				//rectangle(out_img, boxes[j].tl() - Point(3, word_size.height + 3), boxes[j].tl() + Point(word_size.width, 0), Scalar(255, 0, 255), -1);
				////putText将文本从存储区拷贝到屏幕 这里的out_img便是文字识别结果图
				//putText(out_img, words[j], boxes[j].tl() - Point(1, 1), FONT_HERSHEY_SIMPLEX, scale_font, Scalar(128, 255, 128), (int)(3 * scale_font));
				//out_img_segmentation = out_img_segmentation | group_segmentation;
			}

		}

		double time_ocr = ((double)getTickCount() - t_r) * 1000 / getTickFrequency();
		cout << "执行OCR进行文字识别用时 = " << time_ocr << endl;
		total_time_ocr += (time_ocr + time_ocr_initial);

		cout << "总用时 = " << (time_ocr + time_ocr_initial + time_grouping + time_detection) << endl;

		int ii = i + 1;
		stringstream  str_idx;
		str_idx << ii;
		string str = str_idx.str();

		string detect_txt = "D:\\hx\\edgebox-contour-neumann\\neumann_2015train-detection\\res_img_" + str + ".txt";
		ofstream outfile;
		outfile.open(detect_txt, ios::out);

		for (int i = 0; i < (int)temp_boxes.size(); i++)
		{
			outfile << temp_boxes[i].x << "," << temp_boxes[i].y << "," << (temp_boxes[i].x + temp_boxes[i].width) << "," << (temp_boxes[i].y + temp_boxes[i].height) << endl;
		}

		outfile.close();

		/*resize(out_img_detection, out_img_detection, Size(image.cols*scale_img, image.rows*scale_img));
		string str_detection = "D:\\hx\\edgebox-contour-neumann\\neumann_2015train-detection\\res_img_" + str + ".jpg";
		imwrite(str_detection, out_img_detection);*/

		/*string str_recognition = "D:\\hx\\edgebox-contour-neumann\\neumann_2015train-detection\\res_img_" + str + ".jpg";
		resize(out_img,out_img,Size(image.cols*scale_img,image.rows*scale_img));
		imwrite(str_recognition, out_img);*/
		//namedWindow("recognition", WINDOW_NORMAL);
		   //imshow("文字识别结果（ocr后）", out_img);


		/*string detect_txt = "D:\\hx\\edgebox-contour-neumann\\neumann_2011train-detection\\" + str + ".txt";
		ofstream outfile;
		outfile.open(detect_txt, ios::out);

		for (int i = 0; i < (int)nm_boxes.size(); i++)
		{
			outfile << nm_boxes[i].x << "," << nm_boxes[i].y << "," << (nm_boxes[i].x + nm_boxes[i].width) << "," << (nm_boxes[i].y + nm_boxes[i].height) << endl;
		}

		outfile.close();*/
	}
	return 0;
}

size_t min(size_t x, size_t y, size_t z)
{
	return x < y ? min(x, z) : min(y, z);
}

size_t edit_distance(const string& A, const string& B)
{
	size_t NA = A.size();
	size_t NB = B.size();

	vector< vector<size_t> > M(NA + 1, vector<size_t>(NB + 1));

	for (size_t a = 0; a <= NA; ++a)
		M[a][0] = a;

	for (size_t b = 0; b <= NB; ++b)
		M[0][b] = b;

	for (size_t a = 1; a <= NA; ++a)
	for (size_t b = 1; b <= NB; ++b)
	{
		size_t x = M[a - 1][b] + 1;
		size_t y = M[a][b - 1] + 1;
		size_t z = M[a - 1][b - 1] + (A[a - 1] == B[b - 1] ? 0 : 1);
		M[a][b] = min(x, y, z);
	}

	return M[A.size()][B.size()];
}

bool isRepetitive(const string& s)
{
	int count = 0;
	for (int i = 0; i<(int)s.size(); i++)
	{
		if ((s[i] == 'i') ||
			(s[i] == 'l') ||
			(s[i] == 'I'))
			count++;
	}
	if (count >((int)s.size() + 1) / 2)
	{
		return true;
	}
	return false;
}


void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation)
{
	for (int r = 0; r < (int)group.size(); r++)
	{
		ERStat er = regions[group[r][0]][group[r][1]];
		if (er.parent != NULL) // deprecate the root region
		{
			int newMaskVal = 255;
			int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
			floodFill(channels[group[r][0]], segmentation, Point(er.pixel%channels[group[r][0]].cols, er.pixel / channels[group[r][0]].cols),
				Scalar(255), 0, Scalar(er.level), Scalar(0), flags);
		}
	}
}

bool   sort_by_lenght(const string &a, const string &b){ return (a.size() > b.size()); }


