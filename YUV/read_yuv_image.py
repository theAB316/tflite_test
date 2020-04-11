import numpy as np
import cv2

path = "C:\\Users\\imane\\Desktop\\"
size = 1080*1440

def display_image(img):
    cv2.imshow("image_window", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0

def resize(img, shape):
    return cv2.resize(img, dsize=shape, interpolation=cv2.INTER_CUBIC)


def main():

    # test_image = cv2.imread("sample_image.png", cv2.IMREAD_GRAYSCALE)

    # print(test_image.shape)
    # display_image(test_image)
    # test_image = resize(test_image, (550, 900))

    # print(test_image.shape)
    # display_image(test_image)

    # test_image = test_image.astype('float32')
    # test_image /= 255

    # np.savetxt("resized_img.out", test_image)

    # exit()

    with open(path + "test_img.yuv", "rb") as f:
        
        # Read the first frame
        # f.seek((1440*1080*3)//2)
        first_frame = f.read(size)

        # Convert to ndarray
        img = np.frombuffer(first_frame, dtype="uint8")
        img = img.reshape((1080, 1440, 1))

        mat = cv2.CreateMat(1080, 1440, cv2.CV_8UC1)
        
        exit()

        print(img.shape)
        display_image(img)
        img = img[:, 180: -180]

        print(img.shape)
        display_image(img)
        img = resize(img, (256, 256))

        print(img.shape)
        display_image(img)

        # print(img)

        # with open(path+"resized_img.out", "wb") as f:
        #     f.write(img)

if __name__ == "__main__":
    main()




# np.savetxt("img.out", img.astype(int), fmt="%i", delimiter=" ") 

    # u = b''
    # v = b''
    # for i in range(0, (1080*1440)//4):
    #   u += f.read(1)
    #   v += f.read(1)

    # u = np.frombuffer(u, dtype="uint8")
    # u = u.reshape((540, 720))

    # v = np.frombuffer(v, dtype="uint8")
    # v = v.reshape((540, 720))

    # display_image(u)
    # display_image(v)


    # # Read the 60th frame
    # frame_60 = f.read(size)

    # # Convert to ndarray
    # img60 = np.frombuffer(frame_60, dtype="uint8")
    # img60 = img60.reshape((1080, 1440))

    # # Display the image
    # display_image(img60)

    # u = b''
    # v = b''
    # for i in range(0, (1080*1440)//4):
    #   u += f.read(1)
    #   v += f.read(1)

    # u = np.frombuffer(u, dtype="uint8")
    # u = u.reshape((540, 720))

    # v = np.frombuffer(v, dtype="uint8")
    # v = v.reshape((540, 720))

    # display_image(u)
    # display_image(v)


    # params = f.read(50)
    # print(params)
    # exit()



"""
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "tensorflow/lite/optional_debug_tools.h"


int printMat(cv::Mat m, int x, int y);
int displayMat(cv::Mat img);
int INFILE_OBJECT_CHECK(std::ifstream& infile);
int OUTFILE_OBJECT_CHECK(std::ofstream& outfile);

int main(){
    // Create buffer to hold the image bytes that are to be read.
    // Store char buffer as char range is 0-255 (CHAR_MAX) and acts as uint8_t
    
    int rows = 1080;
    int cols = 1440;
    int length = rows*cols;
    char buffer[length];
    
    int resized_rows = 28;
    int resized_cols = 28;
    /*
    // Create file object to read image in as binary
    std::ifstream infile("test_img.yuv", std::ios::in | std::ios::binary);
    INFILE_OBJECT_CHECK(infile);
    
    // Read the Y frame of the image. (First 1080*1440 bytes)
    infile.seekg(0, std::ios::beg);
    infile.read(buffer, length);
    infile.close();
    
    // Create 2D array to hold the image
    int img[rows][cols];
    
    // Copy the buffer to the img array
    int count_buf = 0;
    
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            img[i][j] = buffer[count_buf++];
        }
    }
    
    // Copy the 2D array to opencv::Mat
    cv::Mat Image(rows, cols, CV_32SC1, &img);
    // printMat(Image, 10, 10);
    // displayMat(Image);
    
    
    // Resize img
    std::cout<< "[INFO] Resizing image.\n";
    cv::Size size(resized_rows, resized_cols);
    cv::Mat resizedImage;
    resize(Image, resizedImage, size, CV_INTER_CUBIC);
    
    //displayMat(resizedImage);
    // printMat(resizedImage, 10, 10);

    */
    
    // this is for testing purposes only
    // Read the Y frame of the image. (First 1080*1440 bytes)
    std::ifstream infile("resized_img.out", std::ios::in | std::ios::binary);
    INFILE_OBJECT_CHECK(infile);
    infile.seekg(0, std::ios::beg);
    infile.read(buffer, 28*28);
    infile.close();
    
    // Create 2D array to hold the image
    int img[28][28];
    
    // Copy the buffer to the img array
    int count_buf = 0;
    
    for(int i=0; i<28; i++){
        for(int j=0; j<28; j++){
            img[i][j] = (int)buffer[count_buf++];
        }
    }
    std::cout<<img[i]<<"\n";
    exit(0);
    
    
    /*
    **
    ** Tf-lite inference
    **
    */
    
    // Load the model.
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("mnist.tflite");

    if(!model){
        printf("[ERROR] Failed to mmap model\n");
        exit(0);
    }
    
    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    interpreter->AllocateTensors();
    
    // Check interpreter state
    // tflite::PrintInterpreterState(interpreter.get());
    
    // Create input for tflite interpreter
    /*
    std::vector<std::vector<float>> tensor;
    
    std::size_t vi = 0;
    std::size_t vj = 0;
    
    for(int i=0; i<resized_rows; i++){
        for(int j=0; j<resized_cols; j++){
            int pixelValue = (int) resizedImage.at<int>(i, j);
            // tensor[vi][vj] = pixelValue;
            // std::cout << pixelValue << " "; 
            
            vj++;
        }
        vi++;
    }
    
    
    for(vi=0; vi<resized_rows; vi++){
        for(vj=0; vj<resized_cols; vj++){
            std::cout << tensor[vi][vj] << " ";
        }
        std::cout << "\n";
    }
    */

    
    /*
    float* input = interpreter->typed_input_tensor<float>(0);
    // Dummy input for testing
    *input = 2.0;

    interpreter->Invoke();

    float* output = interpreter->typed_output_tensor<float>(0);

    printf("Result is: %f\n", *output);
    
    */
    return 0;
}


int printMat(cv::Mat m, int x, int y){
    /* 
    ** Prints all the elements in an cv::Mat object. 
    ** Iterates till lengths x, y.
    */
    
    for(int i=0; i<x; i++){
        for(int j=0; j<y; j++){
            int pixelValue = (int) m.at<int>(i, j);
            
            std::cout << pixelValue << " "; 
        }
        printf("\n");
    }
    printf("\n\n");
    
    return 0;        
}

int displayMat(cv::Mat img){
    std::cout<< "[INFO] Displaying image.\n";
    
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    cv::imshow( "Display window", img);
    cv::waitKey(0);
    
    return 0;
}

int INFILE_OBJECT_CHECK(std::ifstream& infile){
    /*
    ** File object check for ifstream and ofstream.
    */
    
    if(infile.fail()){
        std::cout<< "[ERROR] Cannot open ifstream file.\n";
        exit(0);
    }
    
    return 0;
}

int OUTFILE_OBJECT_CHECK(std::ofstream& outfile){
    /*
    ** File object check for ofstream.
    */
    
    if(outfile.fail()){
        std::cout<< "[ERROR] Cannot open ofstream file.\n";
        exit(0);
    }
    
    return 0;
}


"""
