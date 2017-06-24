#include "densecrf.h"
#include "DGM\serialize.h"

// Store the colors we read, so that we can write them again.
std::vector<Vec3b> vPalette;

// Produce a color image from a bunch of labels
template <typename T>
Mat colorize(const Mat &map)
{
	Mat res(map.size(), CV_8UC3);

	for (int y = 0; y < res.rows; y++) {
		const T *pMap = map.ptr<T>(y);
		Vec3b	*pRes = res.ptr<Vec3b>(y);
		for (int x = 0; x < res.cols; x++) 
			pRes[x]= vPalette[pMap[x]];
	}

	return res;
}

// Simple classifier that is 50% certain that the annotation is correct
Mat classify(const Mat &gt, int nStates)
{
	// Certainty that the groundtruth is correct
	const float GT_PROB = 0.55f;

	Mat res(gt.size(), CV_32FC(nStates));
	res.setTo( (1.0f - GT_PROB) / (nStates - 1));
	
	for (int y = 0; y < res.rows; y++) {
		const byte	* pGt  = gt.ptr<byte>(y);
		float		* pRes = res.ptr<float>(y);
		for (int x = 0; x < res.cols; x++) {

			int state = pGt[x];

			//for (int s = 0; s < nStates; s++) pRes[x * nStates + s] = (1.0f - GT_PROB) / (nStates - 1);
			pRes[x * nStates + state] = GT_PROB;
		} // x
	} // y

	return res;
}

void fillPalette(void)
{
	if (!vPalette.empty()) vPalette.clear();
	vPalette.push_back(Vec3b(   0, 0,   255));		
	vPalette.push_back(Vec3b(   0, 128, 255));		
	vPalette.push_back(Vec3b(   0, 255, 255));		
	vPalette.push_back(Vec3b(   0, 255, 128));		
	vPalette.push_back(Vec3b(   0, 255, 0 ));		
	vPalette.push_back(Vec3b( 128, 255, 0 ));		
	vPalette.push_back(Vec3b( 255, 255, 0 ));		
	vPalette.push_back(Vec3b( 255, 128, 0 ));		
	vPalette.push_back(Vec3b( 255, 0,   0 ));		
	vPalette.push_back(Vec3b( 255, 0,   128));		
	vPalette.push_back(Vec3b( 255, 0,   255));		
	vPalette.push_back(Vec3b( 128, 0,   255));		
	vPalette.push_back(Vec3b(   0, 0,   128));		
	vPalette.push_back(Vec3b(   0, 64,  128));		
	vPalette.push_back(Vec3b(   0, 128, 128));		
	vPalette.push_back(Vec3b(   0, 128, 64));		
	vPalette.push_back(Vec3b(   0, 128, 0 ));		
//	vPalette.push_back(Vec3b(  64, 128, 0 ));					
	vPalette.push_back(Vec3b( 128, 128, 0 ));		
	vPalette.push_back(Vec3b( 128,  64, 0 ));		
	vPalette.push_back(Vec3b( 128,   0, 0 ));		
	vPalette.push_back(Vec3b( 128,   0, 64));		
	
}

int main(int argc, char *argv[])
{
//	if (argc < 4) {
//		printf("Usage: %s image annotations output\n", argv[0] );
//		return 1;
//	}
	
	// Number of labels
	const int nStates = 21;
	
	// Load the color image and some crude annotations (which are used in a simple classifier)
	//Mat img = imread(argv[1], 1);
	Mat img = imread("D:\\Data\\EMDS4\\Original_EM_Images\\t4-g02-11.png", 1);
	if (img.empty()) {
		printf("Failed to load image %s\n", argv[1]);
		return 1;
	}
	imshow("Input Image", img);

	//Mat gt = imread(argv[2], 1);
	Mat gt = imread("D:\\Data\\EMDS4\\Ground_Truth_Images\\t4-g02-11.png", 0);
	gt /= 255;
	gt *= 255;
	if (gt.empty()) {
		printf("Failed to load annotations %s\n", argv[2]);
		return 1;
	}
	imshow("Groundtruth", gt);
	
	if (img.cols != gt.cols || img.rows != gt.rows) {
		printf("Annotation size doesn't match image!\n");
		return 1;
	}
	
	int width = img.cols;
	int height = img.rows;

	/////////// Put your own unary classifier here! ///////////
	gt /= 255;
	gt *= 2;
	Mat pot1 = classify(gt, nStates);		// Pot is CV32FC(nStates)
	fillPalette();
	Mat pot = dgm::Serialize::from("D:\\Res\\Potentials\\t4-g02-11.dat");
	
	printf("nChannels = %d\n", pot.channels());
	if (pot.cols == pot1.cols)		printf("cols OK\n");
	if (pot.rows == pot1.rows)		printf("rows OK\n");
	if (pot.type() == pot1.type())	printf("type OK\n");
	dgm::CGraphExt graph(nStates);
	graph.build(pot.size());
	graph.setNodes(pot);
	vec_byte_t optimalDecoding = dgm::CDecode::decode(&graph);
	Mat solution(pot.size(), CV_8UC1, optimalDecoding.data());
	Mat resDGM = colorize<byte>(solution);


	for (int y = 0; y < pot.rows; y++) {
		float		* pPot = pot.ptr<float>(y);
		for (int x = 0; x < pot.cols; x++) {
			for (int s = 0; s < nStates; s++)
				pPot[x * nStates + s] = -logf(pPot[x * nStates + s]);
		} // x
	} // y



	//imshow("Pot", pot);
	///////////////////////////////////////////////////////////
	
	// Setup the CRF model
	DenseCRF2D crf(width, height, nStates);
	
	// Specify the unary potential as an array of size W*H*(#classes)
	// packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ...
	crf.setUnaryEnergy(reinterpret_cast<float *>(pot.data));
	
	// add a color independent term (feature = pixel location 0..W-1, 0..H-1)
	// x_stddev = 3
	// y_stddev = 3
	// weight = 3
	crf.addPairwiseGaussian(3, 3, 3);
	
	// add a color dependent term (feature = xyrgb)
	// x_stddev = 60
	// y_stddev = 60
	// r_stddev = g_stddev = b_stddev = 20
	// weight = 10
	crf.addPairwiseBilateral(60, 60, 20, 20, 20, img.data, 10);
	
	// Do map inference
	short *map = new short[width * height];
	crf.map(100, map);
	
	// Store the result
	Mat res = colorize<short>(Mat(height, width, CV_16UC1, map));
	
	imshow("Result DGB", resDGM);
	imshow("Result", res);
	cvWaitKey();
	
//	imwrite(argv[3], res);
	
	delete[] map;

	return 0;
}
