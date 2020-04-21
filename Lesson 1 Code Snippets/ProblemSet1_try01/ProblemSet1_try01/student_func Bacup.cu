#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
	#ifdef CUDA_ERROR_CHECK
		if (cudaSuccess != err)
		{
			fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
				file, line, cudaGetErrorString(err));
			exit(-1);
		}
	#endif

    return;
}

inline void __cudaCheckError(const char *file, const int line)
{
	#ifdef CUDA_ERROR_CHECK
		cudaError err = cudaGetLastError();
		if (cudaSuccess != err)
		{
			fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
				file, line, cudaGetErrorString(err));
			exit(-1);
		}

		err = cudaDeviceSynchronize();
		if (cudaSuccess != err)
		{
			fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
				file, line, cudaGetErrorString(err));
			exit(-1);
		}
	#endif

    return;
}

__global__ void rgb_2_grey(uchar* const greyImage, const uchar4* const rgbImage, int rows, int columns)
{
    int rgb_x = blockIdx.x * blockDim.x + threadIdx.x; //x coordinate of pixel
    int rgb_y = blockIdx.y * blockDim.y + threadIdx.y; //y coordinate of pixel

    if ((rgb_x >= columns) && (rgb_y >= rows)) {
        return;
    }

    int rgb_ab = rgb_y*columns + rgb_x; //absolute pixel position
    uchar4 rgb_Img = rgbImage[rgb_ab];
    greyImage[rgb_ab] = uchar((float(rgb_Img.x))*0.299f + (float(rgb_Img.y))*0.587f + (float(rgb_Img.z))*0.114f);
}

using namespace cv;
using namespace std;

void Proc_Img(uchar4** h_RGBImage, uchar** h_greyImage, uchar4 **d_RGBImage, uchar** d_greyImage);
void RGB_2_Greyscale(uchar* const d_greyImage, uchar4* const d_RGBImage, size_t num_Rows, size_t num_Cols);
void Save_Img();

Mat img_RGB;
Mat img_Grey;
uchar4 *d_rgbImg;
uchar *d_greyImg; 

int main()
{
        uchar4* h_rgbImg;
        //uchar4* d_rgbImge=0;
        uchar* h_greyImg;
        //uchar* d_greyImge=0;

        Proc_Img(&h_rgbImg, &h_greyImg, &d_rgbImg, &d_greyImg);
        RGB_2_Greyscale(d_greyImg, d_rgbImg, img_RGB.rows, img_RGB.cols);
        Save_Img();

    return 0;
}

void Proc_Img(uchar4** h_RGBImage, uchar** h_greyImage, uchar4 **d_RGBImage, uchar** d_greyImage){
    cudaFree(0);
    CudaCheckError();

    //loads image into a matrix object along with the colors in BGR format (must convert to rgb).
    Mat img = imread("cinque_terre_small.jpg", CV_LOAD_IMAGE_COLOR);
    if (img.empty()){
        cerr << "couldnt open file ..." << "cinque_terre_small.jpg" << endl;
        exit(1);
    }

    //converts color type from BGR to RGB
    cvtColor(img, img_RGB, CV_BGR2RGBA);

    //allocate memory for new greyscale image. 
    //img.rows returns the range of pixels in y, img.cols returns range of pixels in x
    //CV_8UC1 means 8 bit unsigned(non-negative) single channel of color, aka greyscale.
    //all three of the parameters allow the create function in the Mat class to determine how much memory to allocate
    img_Grey.create(img.rows, img.cols, CV_8UC1);

    //creates rgb and greyscale image arrays
    *h_RGBImage = (uchar4*)img_RGB.ptr<uchar>(0); //.ptr is a method in the mat class that returns a pointer to the first element of the matrix.
    *h_greyImage = (uchar*)img_Grey.ptr<uchar>(0);        //this is just like a regular array/pointer mem address to first element of the array. This is templated
                                                          //in this case the compiler runs the function for returning pointer of type unsigned char. for rgb image it is
                                                          //cast to uchar4 struct to hold r,g, and b values.

    const size_t num_pix = (img_RGB.rows) * (img_RGB.cols); //amount of pixels 

    //allocate memory on gpu
    cudaMalloc(d_RGBImage, sizeof(uchar4) * num_pix); //bites of 1 uchar4 times # of pixels gives number of bites necessary for array
    CudaCheckError();
    cudaMalloc(d_greyImage, sizeof(uchar) * num_pix);//bites of uchar times # pixels gives number of bites necessary for array
    CudaCheckError();
    cudaMemset(*d_greyImage, 0, sizeof(uchar) * num_pix);
    CudaCheckError();


    //copy array into allocated space
    cudaMemcpy(*d_RGBImage, *h_RGBImage, sizeof(uchar4)*num_pix, cudaMemcpyHostToDevice);
    CudaCheckError();


    d_rgbImg = *d_RGBImage;
    d_greyImg = *d_greyImage; 
}


void RGB_2_Greyscale(uchar* const d_greyImage, uchar4* const d_RGBImage, size_t num_Rows, size_t num_Cols){

    const int BS = 16;
    const dim3 blockSize(BS, BS);
    const dim3 gridSize((num_Cols / BS) + 1, (num_Rows / BS) + 1); 

    rgb_2_grey <<<gridSize, blockSize>>>(d_greyImage, d_RGBImage, num_Rows, num_Cols);

    cudaDeviceSynchronize(); CudaCheckError();

}



void Save_Img(){

    const size_t num_pix = (img_RGB.rows) * (img_RGB.cols);
    cudaMemcpy(img_Grey.ptr<uchar>(0), d_greyImg, sizeof(uchar)*num_pix, cudaMemcpyDeviceToHost);
    CudaCheckError();


    imwrite("result.jpg", img_Grey);

    cudaFree(d_rgbImg);
    cudaFree(d_greyImg);

}