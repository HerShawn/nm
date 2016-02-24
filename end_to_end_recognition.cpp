

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

//��evaluationʱ����ı༭���������
//Calculate edit distance netween two words
size_t edit_distance(const string& A, const string& B);
size_t min(size_t x, size_t y, size_t z);
bool   isRepetitive(const string& s);
bool   sort_by_lenght(const string &a, const string &b);

//ͼ��
//Draw ER's in an image via floodFill
void   er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation);

//����һ��Ŀ¼����������ļ�
void getFiles(string path, vector<string>& files)
{
	//�ļ����  
	long   hFile = 0;
	//�ļ���Ϣ  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//�����Ŀ¼,����֮  
			//�������,�����б�  
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


//Neuamnn�˶Զ˷���������ڴ�����ȡER��+Neuamnn��ͨ�������׶η���+��֦��������ı��кϲ�+���ñ༭�������ȫ����׼��WRA
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
	char * filePath = "D:\\hx\\edgebox-contour-neumann\\train-textloc";
	////��ȡ��·���µ������ļ�  
	getFiles(filePath, files);

	//char str[30];
	int size = files.size();

	vector<string> file_names;
	vector<string> jpg_names;
	vector<string> txt_names;
	for (int i = 0; i < size; i++)
	{
		string file_name = files[i].c_str();
		//cout << file_name << endl;
		file_names.push_back(file_name);

	}

	string flag = ".";
	size_t sub_pos1 = file_names[1].find_first_of(flag, 0) - 3;
	string ori_names = file_names[1].substr(0, sub_pos1);

	for (int i = 0; i < (size / 2); i++)
	{
		string flag = "jpg";
		size_t sub_pos1 = file_names[i].find(flag, 0) - 4;
		size_t sub_pos2 = 3;

		jpg_names.push_back(ori_names + file_names[i].substr(sub_pos1, sub_pos2) + ".jpg");

		//cout << ori_names + file_names[i].substr(sub_pos1, sub_pos2) + ".jpg" << endl;
		//final_jap_names
	}

	//cout << size << endl;

	for (int i = (size / 2); i <size; i++)
	{
		string flag = "gt_";
		size_t sub_pos1 = file_names[i].find(flag, 0);
		size_t sub_pos2 = 6;

		txt_names.push_back(ori_names + file_names[i].substr(sub_pos1, sub_pos2) + ".txt");

		//cout << ori_names + file_names[i].substr(sub_pos1, sub_pos2) + ".txt" << endl;
		//final_jap_names
	}

	/* �����������Լ��ķ������д���ռ��ı��� */
	if (freopen("dialog.txt", "w", stdout) == NULL)
		fprintf(stderr, "error redirecting stdout\n");

	for (int i = 0; i < (size / 2); i++)
	{


		//argv[1] = jpg_names[i];
		//argv[2] = txt_names[i];

		Mat image;
		if (argc > 1)
			//image = imread(argv[1]);
			image = imread(jpg_names[i]);
		else
		{
			/* cout << "    Usage: " << argv[0] << " <input_image> [<gt_word1> ... <gt_wordN>]" << endl;
			 return(0);*/
			//cout << ori_names << endl;
			return 0;
		}

		cout << endl << endl << "�������е���" << (i + 100) << "��ͼƬ" << endl;
		cout << "ͼƬ��=" << image.cols << endl;
		cout << "ͼƬ��=" << image.rows << endl;

		/*Text Detection*/
		// Extract channels to be processed individually
		vector<Mat> channels;
		Mat grey;
		cvtColor(image, grey, COLOR_RGB2GRAY);
		// Notice here we are only using grey channel, see textdetection.cpp for example with more channels
		channels.push_back(grey);
		channels.push_back(255 - grey);

		//computeNMChannels(image, channels, ERFILTER_NM_IHSGrad);
		//int cn = (int)channels.size();
		//cout << "cn=" << cn << endl;
		//// Append negative channels to detect ER- (bright regions over dark background)
		//for (int c = 0; c < cn - 1; c++)
		//{
		//	channels.push_back(255 - channels[c]);
		//	//cout << "channels.size=" << (int)channels.size() << "c=" << c << endl;
		//}

		//cout << "channels.size=" << (int)channels.size() << endl;

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
		cout << "���ּ����ʱ = " << time_detection << endl;
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
		//�߸�ͨ���ģ�������ergrouping���ﵼ�����ظ���H��255-S�����غϡ��غϵ�����װ��nm_region_groups��nm_boxes���档
		erGrouping(image, channels, regions, nm_region_groups, nm_boxes, ERGROUPING_ORIENTATION_HORIZ);
		double time_grouping = ((double)getTickCount() - t_g) * 1000 / getTickFrequency();
		cout << "�ı��кϲ���ʱ = " << time_grouping << endl;
		total_time_grouping += time_grouping;


		/*Text Recognition (OCR)*/

		double t_r = (double)getTickCount();
		Ptr<OCRTesseract> ocr = OCRTesseract::create();
		double time_ocr_initial = ((double)getTickCount() - t_r) * 1000 / getTickFrequency();
		cout << "����OCR��ʱ = " << time_ocr_initial << endl;
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

			//boundingbox������rectangle�����������ģ������out_img_detection�����ּ��Ľ����
			//���bounding boxֱ�����ı��кϲ�ergrouping�õ���nm_boxes����
			rectangle(out_img_detection, nm_boxes[i].tl(), nm_boxes[i].br(), Scalar(3, 128, 255), 2);

			//
			Mat group_img = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);

			//vector< vector<Vec2i> >& nm_region_groups
			er_draw(channels, regions, nm_region_groups[i], group_img);
			Mat group_segmentation;
			group_img.copyTo(group_segmentation);
			//image(nm_boxes[i]).copyTo(group_img);
			group_img(nm_boxes[i]).copyTo(group_img);
			//copyMakeBorder����ͼ���������߽硣������߽�����
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
				Size word_size = getTextSize(words[j], FONT_HERSHEY_SIMPLEX, (double)scale_font, (int)(3 * scale_font), NULL);
				rectangle(out_img, boxes[j].tl() - Point(3, word_size.height + 3), boxes[j].tl() + Point(word_size.width, 0), Scalar(255, 0, 255), -1);
				//putText���ı��Ӵ洢����������Ļ �����out_img��������ʶ����ͼ
				putText(out_img, words[j], boxes[j].tl() - Point(1, 1), FONT_HERSHEY_SIMPLEX, scale_font, Scalar(128, 255, 128), (int)(3 * scale_font));
				out_img_segmentation = out_img_segmentation | group_segmentation;
			}

		}

		double time_ocr = ((double)getTickCount() - t_r) * 1000 / getTickFrequency();
		cout << "ִ��OCR��������ʶ����ʱ = " << time_ocr << endl;
		total_time_ocr += (time_ocr + time_ocr_initial);

		cout << "����ʱ = " << (time_ocr + time_ocr_initial + time_grouping + time_detection) << endl;
		/* Recognition evaluation with (approximate) hungarian matching and edit distances */




		if (argc > 2)
		{
			int num_gt_characters = 0;
			//vector<string> words_gt;



			//����2
			string a[100];              //���� string ���ͣ���100�е��ı�����Ҫ������ 
			int gt_line_num = 0;
			ifstream infile;

			infile.open(txt_names[i], ios::in);

			while (!infile.eof())            // ��δ���ļ�����һֱѭ�� 
			{
				getline(infile, a[gt_line_num], '\n');//��ȡһ�У��Ի��з����������� a[] ��
				gt_line_num++;                    //��һ��
			}
			//for (int ii = 0; ii<gt_line_num; ii++)        // ��ʾ��ȡ��txt���� 
			//{
			//	cout << a[ii] << endl;
			//}
			infile.close();

			vector<string> words_gt;

			for (int i = 0; i<(gt_line_num-1); i++)
			{
				string flag = "\"";
				size_t sub_pos1 = a[i].find_first_of(flag, 0) + 1;
				size_t sub_pos2 = a[i].size() - 1;

				words_gt.push_back(a[i].substr(sub_pos1, sub_pos2 - sub_pos1));
				//cout << a[i].substr(sub_pos1, sub_pos2 - sub_pos1)<<endl;

				if (words_gt[i].size()>0)
				{
					num_gt_characters += (int)(words_gt[words_gt.size() - 1].size());

				}
			}

			cout << "groud truth�е��ַ����� = " << num_gt_characters << endl;

			//for (int i = 0; i<gt_line_num; i++)
			//      {


			//          string s = string(a[i]);
			//          if (s.size() > 0)
			//          {
			//              words_gt.push_back(string(a[i]));
			//              //cout << " GT word " << words_gt[words_gt.size()-1] << endl;
			//              num_gt_characters += (int)(words_gt[words_gt.size()-1].size());
			//          }
			//      }

			if (words_detection.empty())
			{
				//cout << endl << "number of characters in gt = " << num_gt_characters << endl;
				cout << "TOTAL_EDIT_DISTANCE = " << num_gt_characters << endl;
				cout << "EDIT_DISTANCE_RATIO = 1" << endl;
			}
			else
			{
				//����ֵ�Ӵ�С�������ѣ�����Ҫ��
				sort(words_gt.begin(), words_gt.end(), sort_by_lenght);

				int max_dist = 0;
				vector< vector<int> > assignment_mat;
				for (int i = 0; i < (int)words_gt.size(); i++)
				{
					vector<int> assignment_row(words_detection.size(), 0);
					assignment_mat.push_back(assignment_row);
					for (int j = 0; j < (int)words_detection.size(); j++)
					{
						//����ֵ������ocrʶ�𵥴�������ȡ���༭���롱��
						assignment_mat[i][j] = (int)(edit_distance(words_gt[i], words_detection[j]));
						//max_dist������ֻ������һ��������Ͻ磬������ǿ�����һ������ķ�Χ�ڵ����༭����search_dist��
						max_dist = max(max_dist, assignment_mat[i][j]);
					}
				}

				vector<int> words_detection_matched;

				int total_edit_distance = 0;
				int tp = 0, fp = 0, fn = 0;
				//�༭�����0��ʼ������0��ʾwords_gt[i]��words_detection[min_dist_idx]��ȫƥ�䣬����Ҫ�༭һ���ַ���
				//�༭����Խ�󣬱�����������֮��Ĳ���Խ��Ҳ����ʶ���Ч��Խ�
				for (int search_dist = 0; search_dist <= max_dist; search_dist++)
				{
					//�ӵ�һ����ֵ����words_gt[0]��ʼ�жϣ�ֱ�����һ����ֵ����words_gt[assignment_mat.size()-1];
					for (int i = 0; i < (int)assignment_mat.size(); i++)
					{
						//��Ѱwords_gt[i]��words_detection����������ʶ�𵥴��е���С�༭���룻
						int min_dist_idx = (int)distance(assignment_mat[i].begin(),
							min_element(assignment_mat[i].begin(), assignment_mat[i].end()));
						//����words_gt[i]��words_detection[min_dist_idx]�ı༭���룬����tp\fp\fn\����༭��������ԣ�
						if (assignment_mat[i][min_dist_idx] == search_dist)
						{
							cout << " GT word \"" << words_gt[i] << "\" best match \"" << words_detection[min_dist_idx] << "\" with dist " << assignment_mat[i][min_dist_idx] << endl;
							//����༭����Ϊ0����ô�ж�Ϊ�����ʵ������ģ�tp��һ��
							if (search_dist == 0)
								tp++;
							//����ʶ�𵥴ʺ���ֵ����֮�仹����һ�����ϵ��ַ���ͬ�ģ���ô�ж�Ϊ�����ʵ�Ǽٵģ���fp��һ��ͬʱ
							//����һЩ��ֵ����û�н����ж�Ϊ�棬���fn��һ��
							else { fp++; fn++; }

							//�ۼ�����һ����ֵ���ʺ�ʶ�𵥴ʼ�ı༭���룻
							total_edit_distance += assignment_mat[i][min_dist_idx];
							//words_detection_matched�������ǿ��Ƿ��ʶ���˵��ʣ�������precision
							words_detection_matched.push_back(min_dist_idx);
							//��������ֵɾȥ��������Ϻ󻹲�������ֵ����ûʶ��ģ�������recall;
							words_gt.erase(words_gt.begin() + i);
							assignment_mat.erase(assignment_mat.begin() + i);
							for (int j = 0; j < (int)assignment_mat.size(); j++)
							{
								//��ƥ��ı�Ϊ�������ܺ���������ƥ�䡱,���ͼ�����
								assignment_mat[j][min_dist_idx] = INT_MAX;
							}
							i--;
						}
					}
				}

				//��������ֵ��δʶ�𵽵ģ����Ǳ���Ϊ��Щ���ʲ����������ûʶ�𣬽����˲�ȫ�ʣ�ͬʱ�ۼƱ༭���룻
				for (int j = 0; j < (int)words_gt.size(); j++)
				{
					cout << " GT word \"" << words_gt[j] << "\" no match found" << endl;
					fn++;
					total_edit_distance += (int)words_gt[j].size();
				}
				//������ʶ��ֵ�����Ǳ���Ϊʶ�������ֵ��ʵ�Ǽٵģ������˾�׼�ʣ�ͬʱ�ۼƱ༭���롣
				for (int j = 0; j < (int)words_detection.size(); j++)
				{
					if (find(words_detection_matched.begin(), words_detection_matched.end(), j) == words_detection_matched.end())
					{
						cout << " Detection word \"" << words_detection[j] << "\" no match found" << endl;
						fp++;
						total_edit_distance += (int)words_detection[j].size();
					}
				}


				//cout << endl << "number of characters in gt = " << num_gt_characters << endl;
				
				cout << "TOTAL_EDIT_DISTANCE = " << total_edit_distance << endl;
				float edit_distance_ratio = (float)total_edit_distance / num_gt_characters;
				total_edit_distance_ratio += edit_distance_ratio;
				cout << "EDIT_DISTANCE_RATIO = " << edit_distance_ratio << endl;
				cout << "TP = " << tp << endl;
				cout << "FP = " << fp << endl;
				cout << "FN = " << fn << endl;
				float precision = (float)tp / (tp + fp);
				total_precision += precision;
				cout << "precision= TP/(TP+FP) = " << precision << endl;
				float recall = (float)tp / (tp + fn);
				total_recall +=  recall;
				cout << "recall	  = TP/(TP+FN) = " << recall << endl;

				//	��ȷ�ȣ�Precision����P = TP / (TP + FP);  ��ӳ�˱��������ж������������������������ı���
				//�ٻ���(Recall)��Ҳ��Ϊ True Positive Rate:R = TP / (TP + FN) = 1 - FN / T;  ��ӳ�˱���ȷ�ж�������ռ�ܵ������ı���

			}
		}


		////

		int ii = i + 100;
		stringstream  str_idx;
		str_idx << ii;
		string str = str_idx.str();

		resize(out_img_detection,out_img_detection,Size(image.cols*scale_img,image.rows*scale_img));
		//   imshow("�����кϲ�����Χ���ο�boundingboxֱ����ergrouping��nmbox�ó���", out_img_detection);
		string str_detection ="D:\\hx\\edgebox-contour-neumann\\neumann_detection\\"+ str + ".jpg";
		imwrite(str_detection, out_img_detection);

		/*string str_recognition = "G:\\����\\icdar2011\\train\\train-recognition\\" + str + ".jpg";
		resize(out_img,out_img,Size(image.cols*scale_img,image.rows*scale_img));
		imwrite(str_recognition, out_img);*/
		////namedWindow("recognition", WINDOW_NORMAL);
		//   imshow("����ʶ������ocr��", out_img);


		//imshow("�����кϲ���ͼ������ergrouping��nm_region_groups��floodfillͼ��ó���", out_img_segmentation);
		//string str_decomposition = "G:\\����\\icdar2011\\train\\train-decomposition\\" + str + ".jpg";
		//imwrite(str_ergrouping, out_img_segmentation);
		//imshow("�����кϲ�ǰͼ������δ����ergrouping�����Ƕ�ͨ�������׶η����ֱ��floodfill)", out_img_decomposition);
		//imwrite(str_decomposition, out_img_decomposition);
		////

		////
	
	
	}

	cout << endl << endl << "��������ָ���ǣ�" << endl;
	float avg_precision = (float)total_precision / 229;
	cout << "avg_precision = " << avg_precision << endl;
	float avg_recall = (float)total_recall / 229;
	cout << "avg_recall = " << avg_recall << endl;
	float avg_f_score = (avg_precision/2) + (avg_recall/2);
	cout << "avg_f_score = " << avg_f_score << endl;
	float avg_edit_distance_ratio = (float)total_edit_distance_ratio / 229;
	cout << "avg_edit_distance_ratio = " << avg_edit_distance_ratio <<endl<< endl;


	float avg_time_detection = (float)total_time_detection / 229;
	cout << "avg_time_detection =" << avg_time_detection << endl;
	float avg_time_grouping = (float)total_time_grouping / 229;
	cout << "avg_time_grouping =" << avg_time_grouping << endl;
	float avg_time_ocr = (float)total_time_ocr / 229;
	cout << "avg_time_ocr =" << avg_time_ocr << endl<<endl<<endl;


	//waitKey(0);

	
	/* this output will go to a file */
	printf("icdar2011���ݿ�������.\n");
	/*close the standard output stream*/
	fclose(stdout);
	return 0;

}

size_t min(size_t x, size_t y, size_t z)
{
    return x < y ? min(x,z) : min(y,z);
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
            size_t x = M[a-1][b] + 1;
            size_t y = M[a][b-1] + 1;
            size_t z = M[a-1][b-1] + (A[a-1] == B[b-1] ? 0 : 1);
            M[a][b] = min(x,y,z);
        }

    return M[A.size()][B.size()];
}

bool isRepetitive(const string& s)
{
    int count = 0;
    for (int i=0; i<(int)s.size(); i++)
    {
        if ((s[i] == 'i') ||
                (s[i] == 'l') ||
                (s[i] == 'I'))
            count++;
    }
    if (count > ((int)s.size()+1)/2)
    {
        return true;
    }
    return false;
}


void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation)
{
    for (int r=0; r<(int)group.size(); r++)
    {
        ERStat er = regions[group[r][0]][group[r][1]];
        if (er.parent != NULL) // deprecate the root region
        {
            int newMaskVal = 255;
            int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
            floodFill(channels[group[r][0]],segmentation,Point(er.pixel%channels[group[r][0]].cols,er.pixel/channels[group[r][0]].cols),
                      Scalar(255),0,Scalar(er.level),Scalar(0),flags);
        }
    }
}

bool   sort_by_lenght(const string &a, const string &b){return (a.size()>b.size());}




//����3
//ifstream in(argv[1]);
//string line;
/*vector<string> words;
while (in >> line)
{
words.push_back(line);
}
in.close();
vector<string>::const_iterator it = words.begin();
while (it != words.end())
{
cout << *it << endl;
++it;
}*/
//char s[80];
//in.getline(s, 80);
//cout << s << endl;
//in.close();
//




//����6 ��MFC�����²�����
//CStdioFile file;
//if (!file.Open("d:\\1.txt", CFile::modeRead))
//{
//	AfxMessageBox("can not open file!");
//	return;
//}
//char *a[5];
//int nCount = 0;
//while (file.ReadString(strLine))
//{
//	//CString strLine = "aaa bbb cccc";
//	CString strTemp;
//	char *ss = strLine.GetBufferSetLength(strLine.GetLength());
//	char *p;
//	int nCount = 0;
//	for (p = strtok(ss, " "); p != NULL; p = strtok(NULL, " "))
//	{
//		a[nCount] = p;
//		nCount++;
//	}
//}

////����4
//char c;
//FILE*fp = NULL;//��Ҫע��
//fp = fopen(argv[1], "r");
//if (NULL == fp) return -1;//Ҫ���ش������
//int num = 2;
//while (fscanf(fp, "%c", &c) != EOF)
//{
//	//printf("%c", c);
//	argv[num++] = &c;
//	printf("%c", c);

//}

//����7


//char output[100];
//char c;
//FILE*fp = NULL;//��Ҫע��
//fp = fopen(argv[1], "r");
//if (NULL == fp) return -1;//Ҫ���ش������
//int num1 = 0;
//while (fscanf(fp, "%c", &c) != EOF)
//{

//	output[num1++] = c;

//}

//char seps[] = "\"";
//char *token;
//bool flag = false;
//int num2 = 3;
//cout << output << ' ';
//token = strtok(output, seps);

//while (token != NULL)
//{
//	if (flag == true)
//	{
//		printf("%s\n", token);
//		sprintf(argv[num2++], token);
//	}
//	flag = !flag;
//	token = strtok(NULL, seps);
//}
//