#pragma warning(disable:4996)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cnn.h"

const char* CLASS_NAME[] = {
	"airplane",
	"automobile",
	"bird",
	"cat",
	"deer",
	"dog",
	"frog",
	"horse",
	"ship",
	"truck"
};

void* readfile(const char* filename, int nbytes) {
	void* buf = malloc(nbytes);
	if (buf == NULL) {
		perror("error while malloc");
		exit(1);
	}

	FILE* fp = fopen(filename, "rb");
	if (fp == NULL) {
		perror("error while openeing");
		exit(1);
	}

	int retv = fread(buf, 1, nbytes, fp);
	if (retv != nbytes) {
		perror("error while read");
	}

	if (fclose(fp) != 0) {
		perror("error while closing");
		exit(1);
	}
	return buf;
}

void* read_bytes(const char* fn, size_t n) {
    FILE* f = fopen(fn, "rb");
    void* bytes = malloc(n);
    size_t r = fread(bytes, 1, n, f);
    fclose(f);
    if (r != n) {
        fprintf(stderr,
            "%s: %zd bytes are expected, but %zd bytes are read.\n",
            fn, n, r);
        exit(EXIT_FAILURE);
    }
    return bytes;
}

const int NETWORK_SIZES[] = {
    64 * 3 * 3 * 3, 64,
    64 * 64 * 3 * 3, 64,
    128 * 64 * 3 * 3, 128,
    128 * 128 * 3 * 3, 128,
    256 * 128 * 3 * 3, 256,
    256 * 256 * 3 * 3, 256,
    256 * 256 * 3 * 3, 256,
    512 * 256 * 3 * 3, 512,
    512 * 512 * 3 * 3, 512,
    512 * 512 * 3 * 3, 512,
    512 * 512 * 3 * 3, 512,
    512 * 512 * 3 * 3, 512,
    512 * 512 * 3 * 3, 512,
    512 * 512, 512,
    512 * 512, 512,
    10 * 512, 10
};

float* read_network() {
    return (float*)read_bytes("network.bin", 60980520);
}

float** slice_network(float* p) {
    float** r = (float**)malloc(sizeof(float*) * 32);
    for (int i = 0; i < 32; ++i) {
        r[i] = p;
        p += NETWORK_SIZES[i];
    }
    return r;
}

int main(int argc, char** argv) {
	if (argc != 3) {
		perror("error while get argument");
		exit(1);
	}
	if (strcmp("answer.txt", argv[2]) == 0) {
		perror("'answer.txt' is unauthorized name");
		exit(1);
	}
	int num_of_image = atoi(argv[1]);
	if (num_of_image < 0 || num_of_image > 10000) {
		perror("number of images is 1 to 10000");
		exit(1);
	}
	float* images = (float*)readfile("images.bin", sizeof(float) * 32 * 32 * 3 * num_of_image);
    float* network = read_network();
    float** network_sliced = slice_network(network);
	int* labels = (int*)malloc(sizeof(int) * num_of_image);
	float* confidences = (float*)malloc(sizeof(float) * num_of_image);
	
	cnn_init();
	time_t start, end;
	start = clock();
	//cnn_seq(images, network, labels, confidences, num_of_image);
	cnn(images, network_sliced, labels, confidences, num_of_image);
	end = clock();
	printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLK_TCK);

	int* labels_ans = (int*)readfile("labels.bin", sizeof(int) * num_of_image);
	double acc = 0;

	FILE* fp = fopen(argv[2], "w");
	for (int i = 0; i < num_of_image; ++i) {
		fprintf(fp, "Image %04d : %d : %-10s\t%f\n", i, labels[i], CLASS_NAME[labels[i]], confidences[i]);
		if (labels[i] == labels_ans[i]) ++acc;
	}
	fprintf(fp, "Accuracy: %f\n", acc / num_of_image);
	fclose(fp);
	compare(argv[2], num_of_image);


	free(images);
	free(network);
	free(labels);
	free(confidences);
	free(labels_ans);


	return 0;

}