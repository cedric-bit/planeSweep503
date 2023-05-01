
#include "main.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

#include <iostream>


#define CHK(code) \
do { \
    if ((code) != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s %s %i\n", \
                        cudaGetErrorString((code)), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

#define BLOCKSIZE 16
#define kernel_size 9
#define MI(r, c, width) ((r) * (width) + (c))



__constant__ double const_refpKinv[9];
__constant__ double const_refpRinv[9];
__constant__ double const_refptinv[3];
__constant__ double const_campR[9];
__constant__ double const_campK[9];
__constant__ double const_campt[3];


__global__ void compute_shared_cost(double* refpKinv, double* refpRinv, double* refptinv, double* campR, double* campK, double* campt, uint8_t * in, uint8_t * camera_yuv_0_arr, int width, int height, float* cost_cube)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int kernel_radius = kernel_size / 2;
	const int SHARED_WIDTH = blockDim.x + 2 * kernel_radius;
	const int SHARED_HEIGHT = blockDim.y + 2 * kernel_radius;

	extern __shared__ uint8_t shared_mem[];

	int shared_x = threadIdx.x + kernel_radius;
	int shared_y = threadIdx.y + kernel_radius;
	int shared_idx = shared_y * SHARED_WIDTH + shared_x;

	int image_index = y * width + x;

	if (x < width && y < height)
	{
		shared_mem[shared_idx] = in[MI(y, x, width)];

		// Fill left column of shared memory
		if (threadIdx.x < kernel_radius)
		{
			int left_x = x - kernel_radius;
			int left_shared_x = threadIdx.x;
			int left_shared_idx = MI(shared_y, left_shared_x, SHARED_WIDTH);

			shared_mem[left_shared_idx] = (left_x >= 0) ? in[MI(y, left_x, width)] : 0;
		}

		// Fill right column of shared memory
		if (threadIdx.x >= blockDim.x - kernel_radius)
		{
			int right_x = x + kernel_radius;
			int right_shared_x = threadIdx.x + kernel_radius + 1;
			int right_shared_idx = MI(shared_y, right_shared_x, SHARED_WIDTH);

			shared_mem[right_shared_idx] = (right_x < width) ? in[MI(y, right_x, width)] : 0;
		}

		// Fill top row of shared memory
		if (threadIdx.y < kernel_radius)
		{
			int top_y = y - kernel_radius;
			int top_shared_y = threadIdx.y;
			int top_shared_idx = MI(top_shared_y, shared_x, SHARED_WIDTH);

			shared_mem[top_shared_idx] = (top_y >= 0) ? in[MI(top_y, x, width)] : 0;
		}

		// Fill bottom row of shared memory
		if (threadIdx.y >= blockDim.y - kernel_radius)
		{
			int bottom_y = y + kernel_radius;
			int bottom_shared_y = threadIdx.y + kernel_radius + 1;
			int bottom_shared_idx = MI(bottom_shared_y, shared_x, SHARED_WIDTH);

			shared_mem[bottom_shared_idx] = (bottom_y < height) ? in[MI(bottom_y, x, width)] : 0;
		}

		int tx = threadIdx.x;
		int ty = threadIdx.y;

		// corners
		if (tx < kernel_radius && ty < kernel_radius) {
			// coin superieur gauche
			shared_mem[MI(ty, tx, SHARED_WIDTH)] = (x - kernel_radius < 0 || y - kernel_radius < 0) ? 0 : in[MI(y - kernel_radius, x - kernel_radius, width)];
			// coin superieur droite 
			shared_mem[MI(ty, tx + blockDim.x + kernel_radius, SHARED_WIDTH)] = (x + blockDim.x + kernel_radius >= width || y - kernel_radius < 0) ? 0 : in[MI(y - kernel_radius, x + blockDim.x, width)];
			// coin inferieur gauche

			shared_mem[MI(ty + blockDim.y + kernel_radius, tx, SHARED_WIDTH)] = (x - kernel_radius < 0 || y + blockDim.y + kernel_radius >= height) ? 0 : in[MI(y + blockDim.y, x - kernel_radius, width)];
			// coin inferieru drote
			shared_mem[MI(ty + blockDim.y + kernel_radius, tx + blockDim.x + kernel_radius, SHARED_WIDTH)] = (x + blockDim.x + kernel_radius >= width || y + blockDim.y + kernel_radius >= height) ? 0 : in[MI(y + blockDim.y, x + blockDim.x, width)];
		}

	



	}  __syncthreads();


    if (x >= width || y >= height)
		return;


	const float ZNear = 0.3f;
	const float ZFar = 1.1f;
	const int ZPlanes = 256;

	for (int zi = 0; zi < ZPlanes; zi++)

	{

		// (i) calculate projection index

		// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
		double z = ZNear * ZFar / (ZNear + (((double)zi / (double)ZPlanes) * (ZFar - ZNear)));



		// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
		double X_ref = (refpKinv[0] * x + refpKinv[1] * y + refpKinv[2]) * z;
		double Y_ref = (refpKinv[3] * x + refpKinv[4] * y + refpKinv[5]) * z;
		double Z_ref = (refpKinv[6] * x + refpKinv[7] * y + refpKinv[8]) * z;

		// 3D in ref camera coordinates to 3D world
		double X = refpRinv[0] * X_ref + refpRinv[1] * Y_ref + refpRinv[2] * Z_ref - refptinv[0];
		double Y = refpRinv[3] * X_ref + refpRinv[4] * Y_ref + refpRinv[5] * Z_ref - refptinv[1];
		double Z = refpRinv[6] * X_ref + refpRinv[7] * Y_ref + refpRinv[8] * Z_ref - refptinv[2];

		// 3D world to projected camera 3D coordinates
		double X_proj = campR[0] * X + campR[1] * Y + campR[2] * Z - campt[0];
		double Y_proj = campR[3] * X + campR[4] * Y + campR[5] * Z - campt[1];
		double Z_proj = campR[6] * X + campR[7] * Y + campR[8] * Z - campt[2];

		// Projected camera 3D coordinates to projected camera 2D coordinates
		double x_proj = (campK[0] * X_proj / Z_proj + campK[1] * Y_proj / Z_proj + campK[2]);
		double y_proj = (campK[3] * X_proj / Z_proj + campK[4] * Y_proj / Z_proj + campK[5]);
		double z_proj = Z_proj;

		x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
		y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);


		
		// Calculer le coût
		float cost = 0;
		float cc = 0.0f;
		for (int ky = -kernel_radius; ky <= kernel_radius; ky++)
		{
			for (int kx = -kernel_radius; kx <= kernel_radius; kx++)
			{
				if (x + kx < 0 || x + kx >= width)
					continue;
				if (y + ky < 0 || y + ky >= height)
					continue;
				if (x_proj + kx < 0 || x_proj + kx >= width)
					continue;
				if (y_proj + ky < 0 || y_proj + ky >= height)
					continue;
				int shared_idx = (shared_y + ky) * SHARED_WIDTH + (shared_x + kx);
				int cam_idx = (y_proj + ky) * width + (x_proj + kx);
				float diff = fabsf(shared_mem[shared_idx] - camera_yuv_0_arr[cam_idx]);
				cost += diff;
				cc += 1.0f;
			}
		}
		cost /= cc;

		cost_cube[zi * width * height + image_index] = fminf(cost_cube[width * height * zi + width * y + x], cost);

		__syncthreads();
	}

}



float* compute_shared_cost_lauch(cam const ref, std::vector<cam> const& cam_vector, int window) {

	//int N_threads = 16;
	int wids = ref.width;
	int heig = ref.height;
	// Initialization to MAX value
	float* cost_cube = new float[ref.width * ref.height * ZPlanes];
	for (int i = 0; i < ref.width * ref.height * ZPlanes; ++i)
	{
		cost_cube[i] = 255.f;
	}
	float* dev_cost_cube;
	size_t dev_cost_length = ref.width * ref.height * ZPlanes * sizeof(float);
	CHK(cudaMalloc((void**)&dev_cost_cube, dev_cost_length));
	CHK(cudaMemcpy(dev_cost_cube, cost_cube, dev_cost_length, cudaMemcpyHostToDevice));
	uint8_t* ref_yuv_0_arr = ref.YUV[0].isContinuous() ? ref.YUV[0].data : ref.YUV[0].clone().data;
	size_t length_ref = ref.YUV[0].total() * ref.YUV[0].elemSize();
	int width_ref = ref.YUV[0].cols;

	uint8_t* dev_ref_yuv_0_arr;
	CHK(cudaMalloc((void**)&dev_ref_yuv_0_arr, length_ref * sizeof(uint8_t)));
	CHK(cudaMemcpy(dev_ref_yuv_0_arr, ref_yuv_0_arr, length_ref * sizeof(uint8_t),
		cudaMemcpyHostToDevice));

	double refpKinv[] = { ref.p.K_inv[0], ref.p.K_inv[1], ref.p.K_inv[2], ref.p.K_inv[3], ref.p.K_inv[4], ref.p.K_inv[5], ref.p.K_inv[6], ref.p.K_inv[7], ref.p.K_inv[8] };
	double refpRinv[] = { ref.p.R_inv[0], ref.p.R_inv[1], ref.p.R_inv[2], ref.p.R_inv[3], ref.p.R_inv[4], ref.p.R_inv[5], ref.p.R_inv[6], ref.p.R_inv[7], ref.p.R_inv[8] };
	double refptinv[] = { ref.p.t_inv[0], ref.p.t_inv[1], ref.p.t_inv[2] };

	double* dev_refpKinv, * dev_refpRinv, * dev_refptinv;
	CHK(cudaMalloc((void**)&dev_refpKinv, 9 * sizeof(double)));
	CHK(cudaMalloc((void**)&dev_refpRinv, 9 * sizeof(double)));
	CHK(cudaMalloc((void**)&dev_refptinv, 3 * sizeof(double)));

	CHK(cudaMemcpy(dev_refpKinv, &refpKinv, 9 * sizeof(double),
		cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_refpRinv, &refpRinv, 9 * sizeof(double),
		cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_refptinv, &refptinv, 3 * sizeof(double),
		cudaMemcpyHostToDevice));
	
	int kernel_radius = kernel_size / 2;
	
	for (auto& camera : cam_vector)
	{
		if (camera.name == ref.name)
			continue;   
		

		//std::cout << "Cam: " << camera.name << std::endl;

		uint8_t* camera_yuv_0_arr = camera.YUV[0].isContinuous() ? camera.YUV[0].data : camera.YUV[0].clone().data;
		size_t length_camera = camera.YUV[0].total() * camera.YUV[0].elemSize();
		int width_camera = camera.YUV[0].cols;

		uint8_t* dev_camera_yuv_0_arr;
		CHK(cudaMalloc((void**)&dev_camera_yuv_0_arr, length_camera * sizeof(uint8_t)));
		CHK(cudaMemcpy(dev_camera_yuv_0_arr, camera_yuv_0_arr, length_camera * sizeof(uint8_t),
			cudaMemcpyHostToDevice));

		double campR[] = { camera.p.R[0], camera.p.R[1] , camera.p.R[2] , camera.p.R[3] , camera.p.R[4] , camera.p.R[5] , camera.p.R[6] , camera.p.R[7] , camera.p.R[8] };
		double campK[] = { camera.p.K[0], camera.p.K[1] , camera.p.K[2] , camera.p.K[3] , camera.p.K[4] , camera.p.K[5] };
		double campt[] = { camera.p.t[0], camera.p.t[1] , camera.p.t[2] };

		double* dev_campR, * dev_campK, * dev_campt;
		CHK(cudaMalloc((void**)&dev_campR, 9 * sizeof(double)));
		CHK(cudaMalloc((void**)&dev_campK, 6 * sizeof(double)));
		CHK(cudaMalloc((void**)&dev_campt, 3 * sizeof(double)));

		CHK(cudaMemcpy(dev_campR, &campR, 9 * sizeof(double),
			cudaMemcpyHostToDevice));
		CHK(cudaMemcpy(dev_campK, &campK, 6 * sizeof(double),
			cudaMemcpyHostToDevice));
		CHK(cudaMemcpy(dev_campt, &campt, 3 * sizeof(double),
			cudaMemcpyHostToDevice));

		cam* dev_ref, * dev_camera;



		CHK(cudaMalloc((void**)&dev_ref, sizeof(cam)));
		CHK(cudaMalloc((void**)&dev_camera, sizeof(cam)));

		CHK(cudaMemcpy(dev_ref, &ref, sizeof(cam),
			cudaMemcpyHostToDevice));

		CHK(cudaMemcpy(dev_camera, &camera, sizeof(cam),
			cudaMemcpyHostToDevice));
		// Définition des dimensions des blocs et des grilles
		int block_size_x = 32; // Nombre de threads par bloc dans la direction x
		int block_size_y = 32; // Nombre de threads par bloc dans la direction y
		dim3 block_dim(block_size_x, block_size_y);
		dim3 grid_dim((ref.width + block_size_x - 1) / block_size_x, (ref.height + block_size_y - 1) / block_size_y);

		// Taille de la mémoire partagée (extern __shared__)
		int shared_mem_size = (block_size_x + 2 * kernel_radius) * (block_size_y + 2 * kernel_radius) * sizeof(float);
		//dim3 block_size = dim3(Nx / thread_size.x, Ny / thread_size.y, 1); //(10,1080,1) 
		compute_shared_cost << <grid_dim, block_dim, shared_mem_size >> > (dev_refpKinv, dev_refpRinv, dev_refptinv,
			dev_campR, dev_campK, dev_campt, dev_ref_yuv_0_arr, dev_camera_yuv_0_arr, wids, heig, dev_cost_cube);
		CHK(cudaMemcpy(cost_cube, dev_cost_cube, dev_cost_length,
			cudaMemcpyDeviceToHost));
		CHK(cudaDeviceSynchronize());

		cudaFree(dev_camera_yuv_0_arr);
		cudaFree(dev_campt);
		cudaFree(dev_campK);
		cudaFree(dev_campR);


	}
Error:
	cudaFree(dev_cost_cube);
	cudaFree(dev_refpKinv);
	cudaFree(dev_refpRinv);
	cudaFree(dev_refptinv);
	cudaFree(dev_ref_yuv_0_arr);




	return cost_cube;

}


__global__ void compute_naive_cost(double* refpKinv, double* refpRinv, double* refptinv, double* campR, double* campK, double* campt, int window,
	uint8_t* ref_yuv_0_arr, uint8_t* camera_yuv_0_arr, int width, int height, float* cost_cube)

{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= width || y >= height)
		return;


	const float ZNear = 0.3f;
	const float ZFar = 1.1f;
	const int ZPlanes = 256;

	for (int zi = 0; zi < ZPlanes; zi++)

	{

		// (i) calculate projection index

		// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
		double z = ZNear * ZFar / (ZNear + (((double)zi / (double)ZPlanes) * (ZFar - ZNear)));

		// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
		double X_ref = (refpKinv[0] * x + refpKinv[1] * y + refpKinv[2]) * z;
		double Y_ref = (refpKinv[3] * x + refpKinv[4] * y + refpKinv[5]) * z;
		double Z_ref = (refpKinv[6] * x + refpKinv[7] * y + refpKinv[8]) * z;

		// 3D in ref camera coordinates to 3D world
		double X = refpRinv[0] * X_ref + refpRinv[1] * Y_ref + refpRinv[2] * Z_ref - refptinv[0];
		double Y = refpRinv[3] * X_ref + refpRinv[4] * Y_ref + refpRinv[5] * Z_ref - refptinv[1];
		double Z = refpRinv[6] * X_ref + refpRinv[7] * Y_ref + refpRinv[8] * Z_ref - refptinv[2];

		// 3D world to projected camera 3D coordinates
		double X_proj = campR[0] * X + campR[1] * Y + campR[2] * Z - campt[0];
		double Y_proj = campR[3] * X + campR[4] * Y + campR[5] * Z - campt[1];
		double Z_proj = campR[6] * X + campR[7] * Y + campR[8] * Z - campt[2];

		// Projected camera 3D coordinates to projected camera 2D coordinates
		double x_proj = (campK[0] * X_proj / Z_proj + campK[1] * Y_proj / Z_proj + campK[2]);
		double y_proj = (campK[3] * X_proj / Z_proj + campK[4] * Y_proj / Z_proj + campK[5]);
		double z_proj = Z_proj;

		x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
		y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);

		float cost = 0.0f;
		float cc = 0.0f;
		for (int k = -window / 2; k <= window / 2; k++)
		{
			for (int l = -window / 2; l <= window / 2; l++)
			{
				if (x + l < 0 || x + l >= width)
					continue;
				if (y + k < 0 || y + k >= height)
					continue;
				if (x_proj + l < 0 || x_proj + l >= width)
					continue;
				if (y_proj + k < 0 || y_proj + k >= height)
					continue;

				// Y
				//cost += (int)x_proj;
				cost += fabsf(ref_yuv_0_arr[width * (y + k) + (x + l)] - camera_yuv_0_arr[width * ((int)y_proj + k) + ((int)x_proj + l)]);

				cc += 1.0f;
			}
		}
		cost /= cc;

		//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
				// only the minimum cost for all the cameras is stored

		cost_cube[width * height * zi + width * y + x] = fminf(cost_cube[width * height * zi + width * y + x], cost);
	}


}



float* compute_naive_cost_lauch(cam const ref, std::vector<cam> const& cam_vector, int window) {

	int N_threads = 32;
	int wids = ref.width;
	int heig = ref.height;
	// Initialization to MAX value
	float* cost_cube = new float[ref.width * ref.height * ZPlanes];
	for (int i = 0; i < ref.width * ref.height * ZPlanes; ++i)
	{
		cost_cube[i] = 255.f;
	}
	float* dev_cost_cube;
	size_t dev_cost_length = ref.width * ref.height * ZPlanes * sizeof(float);
	CHK(cudaMalloc((void**)&dev_cost_cube, dev_cost_length));
	CHK(cudaMemcpy(dev_cost_cube, cost_cube, dev_cost_length, cudaMemcpyHostToDevice));
	uint8_t* ref_yuv_0_arr = ref.YUV[0].isContinuous() ? ref.YUV[0].data : ref.YUV[0].clone().data;
	size_t length_ref = ref.YUV[0].total() * ref.YUV[0].elemSize();
	int width_ref = ref.YUV[0].cols;

	uint8_t* dev_ref_yuv_0_arr;
	CHK(cudaMalloc((void**)&dev_ref_yuv_0_arr, length_ref * sizeof(uint8_t)));
	CHK(cudaMemcpy(dev_ref_yuv_0_arr, ref_yuv_0_arr, length_ref * sizeof(uint8_t),
		cudaMemcpyHostToDevice));

	double refpKinv[] = { ref.p.K_inv[0], ref.p.K_inv[1], ref.p.K_inv[2], ref.p.K_inv[3], ref.p.K_inv[4], ref.p.K_inv[5], ref.p.K_inv[6], ref.p.K_inv[7], ref.p.K_inv[8] };
	double refpRinv[] = { ref.p.R_inv[0], ref.p.R_inv[1], ref.p.R_inv[2], ref.p.R_inv[3], ref.p.R_inv[4], ref.p.R_inv[5], ref.p.R_inv[6], ref.p.R_inv[7], ref.p.R_inv[8] };
	double refptinv[] = { ref.p.t_inv[0], ref.p.t_inv[1], ref.p.t_inv[2] };

	double* dev_refpKinv, * dev_refpRinv, * dev_refptinv;
	CHK(cudaMalloc((void**)&dev_refpKinv, 9 * sizeof(double)));
	CHK(cudaMalloc((void**)&dev_refpRinv, 9 * sizeof(double)));
	CHK(cudaMalloc((void**)&dev_refptinv, 3 * sizeof(double)));

	CHK(cudaMemcpy(dev_refpKinv, &refpKinv, 9 * sizeof(double),
		cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_refpRinv, &refpRinv, 9 * sizeof(double),
		cudaMemcpyHostToDevice));
	CHK(cudaMemcpy(dev_refptinv, &refptinv, 3 * sizeof(double),
		cudaMemcpyHostToDevice));
	dim3 thread_size = dim3(N_threads, N_threads); // 192 threads

	dim3 block_size = dim3((ref.width + (thread_size.x - 1)) / thread_size.x,
		(ref.height + (thread_size.y - 1)) / thread_size.y, 1);

	for (auto& camera : cam_vector)
	{
		if (camera.name == ref.name)
			continue;

		//std::cout << "Cam: " << camera.name << std::endl;

		uint8_t* camera_yuv_0_arr = camera.YUV[0].isContinuous() ? camera.YUV[0].data : camera.YUV[0].clone().data;
		size_t length_camera = camera.YUV[0].total() * camera.YUV[0].elemSize();
		int width_camera = camera.YUV[0].cols;

		uint8_t* dev_camera_yuv_0_arr;
		CHK(cudaMalloc((void**)&dev_camera_yuv_0_arr, length_camera * sizeof(uint8_t)));
		CHK(cudaMemcpy(dev_camera_yuv_0_arr, camera_yuv_0_arr, length_camera * sizeof(uint8_t),
			cudaMemcpyHostToDevice));

		double campR[] = { camera.p.R[0], camera.p.R[1] , camera.p.R[2] , camera.p.R[3] , camera.p.R[4] , camera.p.R[5] , camera.p.R[6] , camera.p.R[7] , camera.p.R[8] };
		double campK[] = { camera.p.K[0], camera.p.K[1] , camera.p.K[2] , camera.p.K[3] , camera.p.K[4] , camera.p.K[5] };
		double campt[] = { camera.p.t[0], camera.p.t[1] , camera.p.t[2] };

		double* dev_campR, * dev_campK, * dev_campt;
		CHK(cudaMalloc((void**)&dev_campR, 9 * sizeof(double)));
		CHK(cudaMalloc((void**)&dev_campK, 6 * sizeof(double)));
		CHK(cudaMalloc((void**)&dev_campt, 3 * sizeof(double)));

		CHK(cudaMemcpy(dev_campR, &campR, 9 * sizeof(double),
			cudaMemcpyHostToDevice));
		CHK(cudaMemcpy(dev_campK, &campK, 6 * sizeof(double),
			cudaMemcpyHostToDevice));
		CHK(cudaMemcpy(dev_campt, &campt, 3 * sizeof(double),
			cudaMemcpyHostToDevice));

		cam* dev_ref, * dev_camera;



		CHK(cudaMalloc((void**)&dev_ref, sizeof(cam)));
		CHK(cudaMalloc((void**)&dev_camera, sizeof(cam)));

		CHK(cudaMemcpy(dev_ref, &ref, sizeof(cam),
			cudaMemcpyHostToDevice));

		CHK(cudaMemcpy(dev_camera, &camera, sizeof(cam),
			cudaMemcpyHostToDevice));


		//dim3 block_size = dim3(Nx / thread_size.x, Ny / thread_size.y, 1); //(10,1080,1) 
		compute_naive_cost << <block_size, thread_size >> > (dev_refpKinv, dev_refpRinv, dev_refptinv,
			dev_campR, dev_campK, dev_campt, window, dev_ref_yuv_0_arr, dev_camera_yuv_0_arr, wids, heig, dev_cost_cube);
		CHK(cudaMemcpy(cost_cube, dev_cost_cube, dev_cost_length,
			cudaMemcpyDeviceToHost));
		CHK(cudaDeviceSynchronize());

		cudaFree(dev_camera_yuv_0_arr);
		cudaFree(dev_campt);
		cudaFree(dev_campK);
		cudaFree(dev_campR);


	}
Error:
	cudaFree(dev_cost_cube);
	cudaFree(dev_refpKinv);
	cudaFree(dev_refpRinv);
	cudaFree(dev_refptinv);
	cudaFree(dev_ref_yuv_0_arr);




	return cost_cube;

}





__global__ void compute_naive_constante(int window, uint8_t* ref_yuv_0_arr, uint8_t* camera_yuv_0_arr, int width, int height, float* cost_cube)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	const float ZNear = 0.3f;
	const float ZFar = 1.1f;
	const int ZPlanes = 256;

	for (int zi = 0; zi < ZPlanes; zi++)
	{
		double z = ZNear * ZFar / (ZNear + (((double)zi / (double)ZPlanes) * (ZFar - ZNear)));

		double X_ref = (const_refpKinv[0] * x + const_refpKinv[1] * y + const_refpKinv[2]) * z;
		double Y_ref = (const_refpKinv[3] * x + const_refpKinv[4] * y + const_refpKinv[5]) * z;
		double Z_ref = (const_refpKinv[6] * x + const_refpKinv[7] * y + const_refpKinv[8]) * z;

		double X = const_refpRinv[0] * X_ref + const_refpRinv[1] * Y_ref + const_refpRinv[2] * Z_ref - const_refptinv[0];
		double Y = const_refpRinv[3] * X_ref + const_refpRinv[4] * Y_ref + const_refpRinv[5] * Z_ref - const_refptinv[1];
		double Z = const_refpRinv[6] * X_ref + const_refpRinv[7] * Y_ref + const_refpRinv[8] * Z_ref - const_refptinv[2];

		double X_proj = const_campR[0] * X + const_campR[1] * Y + const_campR[2] * Z - const_campt[0];
		double Y_proj = const_campR[3] * X + const_campR[4] * Y + const_campR[5] * Z - const_campt[1];
		double Z_proj = const_campR[6] * X + const_campR[7] * Y + const_campR[8] * Z - const_campt[2];

		double x_proj = (const_campK[0] * X_proj / Z_proj + const_campK[1] * Y_proj / Z_proj + const_campK[2]);
		double y_proj = (const_campK[3] * X_proj / Z_proj + const_campK[4] * Y_proj / Z_proj + const_campK[5]);
		double z_proj = Z_proj;

		x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
		y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);

		float cost = 0.0f;
		float cc = 0.0f;
		for (int k = -window / 2; k <= window / 2; k++)
		{
			for (int l = -window / 2; l <= window / 2; l++)
			{
				if (x + l < 0 || x + l >= width)
					continue;
				if (y + k < 0 || y + k >= height)
					continue;
				if (x_proj + l < 0 || x_proj + l >= width)
					continue;
				if (y_proj + k < 0 || y_proj + k >= height)
					continue;

				// Y
				//cost += (int)x_proj;
				cost += fabsf(ref_yuv_0_arr[width * (y + k) + (x + l)] - camera_yuv_0_arr[width * ((int)y_proj + k) + ((int)x_proj + l)]);

				cc += 1.0f;
			}
		}
		cost /= cc;

		//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
				// only the minimum cost for all the cameras is stored

		cost_cube[width * height * zi + width * y + x] = fminf(cost_cube[width * height * zi + width * y + x], cost);
	}


}



float* compute_naive_cost_float2D_lauch(cam const ref, std::vector<cam> const& cam_vector, int window) {

	int N_threads = 32;
	int wids = ref.width;
	int heig = ref.height;
	// Initialization to MAX value
	float* cost_cube = new float[ref.width * ref.height * ZPlanes];
	for (int i = 0; i < ref.width * ref.height * ZPlanes; ++i)
	{
		cost_cube[i] = 255.f;
	}
	float* dev_cost_cube;
	size_t dev_cost_length = ref.width * ref.height * ZPlanes * sizeof(float);
	CHK(cudaMalloc((void**)&dev_cost_cube, dev_cost_length));
	CHK(cudaMemcpy(dev_cost_cube, cost_cube, dev_cost_length, cudaMemcpyHostToDevice));
	uint8_t* ref_yuv_0_arr = ref.YUV[0].isContinuous() ? ref.YUV[0].data : ref.YUV[0].clone().data;
	size_t length_ref = ref.YUV[0].total() * ref.YUV[0].elemSize();
	int width_ref = ref.YUV[0].cols;

	uint8_t* dev_ref_yuv_0_arr;
	CHK(cudaMalloc((void**)&dev_ref_yuv_0_arr, length_ref * sizeof(uint8_t)));
	CHK(cudaMemcpy(dev_ref_yuv_0_arr, ref_yuv_0_arr, length_ref * sizeof(uint8_t),
		cudaMemcpyHostToDevice));

	double refpKinv[] = { ref.p.K_inv[0], ref.p.K_inv[1], ref.p.K_inv[2], ref.p.K_inv[3], ref.p.K_inv[4], ref.p.K_inv[5], ref.p.K_inv[6], ref.p.K_inv[7], ref.p.K_inv[8] };
	double refpRinv[] = { ref.p.R_inv[0], ref.p.R_inv[1], ref.p.R_inv[2], ref.p.R_inv[3], ref.p.R_inv[4], ref.p.R_inv[5], ref.p.R_inv[6], ref.p.R_inv[7], ref.p.R_inv[8] };
	double refptinv[] = { ref.p.t_inv[0], ref.p.t_inv[1], ref.p.t_inv[2] };

	cudaMemcpyToSymbol(const_refpKinv, refpKinv, 9 * sizeof(double));
	cudaMemcpyToSymbol(const_refpRinv, refpRinv, 9 * sizeof(double));
	cudaMemcpyToSymbol(const_refptinv, refptinv, 3 * sizeof(double));

	dim3 thread_size = dim3(N_threads, N_threads); // 192 threads

	dim3 block_size = dim3((ref.width + (thread_size.x - 1)) / thread_size.x,
		(ref.height + (thread_size.y - 1)) / thread_size.y, 1);

	for (auto& camera : cam_vector)
	{
		if (camera.name == ref.name)
			continue;

		//std::cout << "Cam: " << camera.name << std::endl;

		uint8_t* camera_yuv_0_arr = camera.YUV[0].isContinuous() ? camera.YUV[0].data : camera.YUV[0].clone().data;
		size_t length_camera = camera.YUV[0].total() * camera.YUV[0].elemSize();
		int width_camera = camera.YUV[0].cols;

		uint8_t* dev_camera_yuv_0_arr;
		CHK(cudaMalloc((void**)&dev_camera_yuv_0_arr, length_camera * sizeof(uint8_t)));
		CHK(cudaMemcpy(dev_camera_yuv_0_arr, camera_yuv_0_arr, length_camera * sizeof(uint8_t),
			cudaMemcpyHostToDevice));

		double campR[] = { camera.p.R[0], camera.p.R[1] , camera.p.R[2] , camera.p.R[3] , camera.p.R[4] , camera.p.R[5] , camera.p.R[6] , camera.p.R[7] , camera.p.R[8] };
		double campK[] = { camera.p.K[0], camera.p.K[1] , camera.p.K[2] , camera.p.K[3] , camera.p.K[4] , camera.p.K[5] };
		double campt[] = { camera.p.t[0], camera.p.t[1] , camera.p.t[2] };

		cudaMemcpyToSymbol(const_campR, campR, 9 * sizeof(double));
		cudaMemcpyToSymbol(const_campK, campK, 6 * sizeof(double));
		cudaMemcpyToSymbol(const_campt, campt, 3 * sizeof(double));

		cam* dev_ref, * dev_camera;



		CHK(cudaMalloc((void**)&dev_ref, sizeof(cam)));
		CHK(cudaMalloc((void**)&dev_camera, sizeof(cam)));

		CHK(cudaMemcpy(dev_ref, &ref, sizeof(cam),
			cudaMemcpyHostToDevice));

		CHK(cudaMemcpy(dev_camera, &camera, sizeof(cam),
			cudaMemcpyHostToDevice));


		//dim3 block_size = dim3(Nx / thread_size.x, Ny / thread_size.y, 1); //(10,1080,1) 
		compute_naive_constante << <block_size, thread_size >> > (window, dev_ref_yuv_0_arr, dev_camera_yuv_0_arr, wids, heig, dev_cost_cube);
		CHK(cudaMemcpy(cost_cube, dev_cost_cube, dev_cost_length,
			cudaMemcpyDeviceToHost));
		CHK(cudaDeviceSynchronize());

		cudaFree(dev_camera_yuv_0_arr);


	}
Error:
	cudaFree(dev_cost_cube);
	cudaFree(dev_ref_yuv_0_arr);




	return cost_cube;

}



__global__ void compute_shared_constante(uint8_t* in, uint8_t* camera_yuv_0_arr, int width, int height, float* cost_cube)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int kernel_radius = kernel_size / 2;
	const int SHARED_WIDTH = blockDim.x + 2 * kernel_radius;
	const int SHARED_HEIGHT = blockDim.y + 2 * kernel_radius;

	extern __shared__ uint8_t shared_mem[];

	int shared_x = threadIdx.x + kernel_radius;
	int shared_y = threadIdx.y + kernel_radius;
	int shared_idx = shared_y * SHARED_WIDTH + shared_x;

	int image_index = y * width + x;

	if (x < width && y < height)
	{
		shared_mem[shared_idx] = in[MI(y, x, width)];

		// Fill left column of shared memory
		if (threadIdx.x < kernel_radius)
		{
			int left_x = x - kernel_radius;
			int left_shared_x = threadIdx.x;
			int left_shared_idx = MI(shared_y, left_shared_x, SHARED_WIDTH);

			shared_mem[left_shared_idx] = (left_x >= 0) ? in[MI(y, left_x, width)] : 0;
		}

		// Fill right column of shared memory
		if (threadIdx.x >= blockDim.x - kernel_radius)
		{
			int right_x = x + kernel_radius;
			int right_shared_x = threadIdx.x + kernel_radius + 1;
			int right_shared_idx = MI(shared_y, right_shared_x, SHARED_WIDTH);

			shared_mem[right_shared_idx] = (right_x < width) ? in[MI(y, right_x, width)] : 0;
		}

		// Fill top row of shared memory
		if (threadIdx.y < kernel_radius)
		{
			int top_y = y - kernel_radius;
			int top_shared_y = threadIdx.y;
			int top_shared_idx = MI(top_shared_y, shared_x, SHARED_WIDTH);

			shared_mem[top_shared_idx] = (top_y >= 0) ? in[MI(top_y, x, width)] : 0;
		}

		// Fill bottom row of shared memory
		if (threadIdx.y >= blockDim.y - kernel_radius)
		{
			int bottom_y = y + kernel_radius;
			int bottom_shared_y = threadIdx.y + kernel_radius + 1;
			int bottom_shared_idx = MI(bottom_shared_y, shared_x, SHARED_WIDTH);

			shared_mem[bottom_shared_idx] = (bottom_y < height) ? in[MI(bottom_y, x, width)] : 0;
		}

		int tx = threadIdx.x;
		int ty = threadIdx.y;

		// corners
		if (tx < kernel_radius && ty < kernel_radius) {
			// coin superieur gauche
			shared_mem[MI(ty, tx, SHARED_WIDTH)] = (x - kernel_radius < 0 || y - kernel_radius < 0) ? 0 : in[MI(y - kernel_radius, x - kernel_radius, width)];
			// coin superieur droite 
			shared_mem[MI(ty, tx + blockDim.x + kernel_radius, SHARED_WIDTH)] = (x + blockDim.x + kernel_radius >= width || y - kernel_radius < 0) ? 0 : in[MI(y - kernel_radius, x + blockDim.x, width)];
			// coin inferieur gauche

			shared_mem[MI(ty + blockDim.y + kernel_radius, tx, SHARED_WIDTH)] = (x - kernel_radius < 0 || y + blockDim.y + kernel_radius >= height) ? 0 : in[MI(y + blockDim.y, x - kernel_radius, width)];
			// coin inferieru drote
			shared_mem[MI(ty + blockDim.y + kernel_radius, tx + blockDim.x + kernel_radius, SHARED_WIDTH)] = (x + blockDim.x + kernel_radius >= width || y + blockDim.y + kernel_radius >= height) ? 0 : in[MI(y + blockDim.y, x + blockDim.x, width)];
		}





	}  __syncthreads();


	if (x >= width || y >= height)
		return;
	const float ZNear = 0.3f;
	const float ZFar = 1.1f;
	const int ZPlanes = 256;

	for (int zi = 0; zi < ZPlanes; zi++)
	{
		double z = ZNear * ZFar / (ZNear + (((double)zi / (double)ZPlanes) * (ZFar - ZNear)));

		double X_ref = (const_refpKinv[0] * x + const_refpKinv[1] * y + const_refpKinv[2]) * z;
		double Y_ref = (const_refpKinv[3] * x + const_refpKinv[4] * y + const_refpKinv[5]) * z;
		double Z_ref = (const_refpKinv[6] * x + const_refpKinv[7] * y + const_refpKinv[8]) * z;

		double X = const_refpRinv[0] * X_ref + const_refpRinv[1] * Y_ref + const_refpRinv[2] * Z_ref - const_refptinv[0];
		double Y = const_refpRinv[3] * X_ref + const_refpRinv[4] * Y_ref + const_refpRinv[5] * Z_ref - const_refptinv[1];
		double Z = const_refpRinv[6] * X_ref + const_refpRinv[7] * Y_ref + const_refpRinv[8] * Z_ref - const_refptinv[2];

		double X_proj = const_campR[0] * X + const_campR[1] * Y + const_campR[2] * Z - const_campt[0];
		double Y_proj = const_campR[3] * X + const_campR[4] * Y + const_campR[5] * Z - const_campt[1];
		double Z_proj = const_campR[6] * X + const_campR[7] * Y + const_campR[8] * Z - const_campt[2];

		double x_proj = (const_campK[0] * X_proj / Z_proj + const_campK[1] * Y_proj / Z_proj + const_campK[2]);
		double y_proj = (const_campK[3] * X_proj / Z_proj + const_campK[4] * Y_proj / Z_proj + const_campK[5]);
		double z_proj = Z_proj;

		x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
		y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);


		// Calculer le coût
		float cost = 0;
		float cc = 0.0f;
		for (int ky = -kernel_radius; ky <= kernel_radius; ky++)
		{
			for (int kx = -kernel_radius; kx <= kernel_radius; kx++)
			{
				if (x + kx < 0 || x + kx >= width)
					continue;
				if (y + ky < 0 || y + ky >= height)
					continue;
				if (x_proj + kx < 0 || x_proj + kx >= width)
					continue;
				if (y_proj + ky < 0 || y_proj + ky >= height)
					continue;
				int shared_idx = (shared_y + ky) * SHARED_WIDTH + (shared_x + kx);
				int cam_idx = (y_proj + ky) * width + (x_proj + kx);
				float diff = fabsf(shared_mem[shared_idx] - camera_yuv_0_arr[cam_idx]);
				cost += diff;
				cc += 1.0f;
			}
		}
		cost /= cc;

		cost_cube[zi * width * height + image_index] = fminf(cost_cube[width * height * zi + width * y + x], cost);

		__syncthreads();
	}

}



float* compute_shared_constante_(cam const ref, std::vector<cam> const& cam_vector, int window) {

	int N_threads = 32;
	int wids = ref.width;
	int heig = ref.height;
	// Initialization to MAX value

	int kernel_radius = kernel_size / 2;
	float* cost_cube = new float[ref.width * ref.height * ZPlanes];
	for (int i = 0; i < ref.width * ref.height * ZPlanes; ++i)
	{
		cost_cube[i] = 255.f;
	}
	float* dev_cost_cube;
	size_t dev_cost_length = ref.width * ref.height * ZPlanes * sizeof(float);
	CHK(cudaMalloc((void**)&dev_cost_cube, dev_cost_length));
	CHK(cudaMemcpy(dev_cost_cube, cost_cube, dev_cost_length, cudaMemcpyHostToDevice));
	uint8_t* ref_yuv_0_arr = ref.YUV[0].isContinuous() ? ref.YUV[0].data : ref.YUV[0].clone().data;
	size_t length_ref = ref.YUV[0].total() * ref.YUV[0].elemSize();
	int width_ref = ref.YUV[0].cols;

	uint8_t* dev_ref_yuv_0_arr;
	CHK(cudaMalloc((void**)&dev_ref_yuv_0_arr, length_ref * sizeof(uint8_t)));
	CHK(cudaMemcpy(dev_ref_yuv_0_arr, ref_yuv_0_arr, length_ref * sizeof(uint8_t),
		cudaMemcpyHostToDevice));

	double refpKinv[] = { ref.p.K_inv[0], ref.p.K_inv[1], ref.p.K_inv[2], ref.p.K_inv[3], ref.p.K_inv[4], ref.p.K_inv[5], ref.p.K_inv[6], ref.p.K_inv[7], ref.p.K_inv[8] };
	double refpRinv[] = { ref.p.R_inv[0], ref.p.R_inv[1], ref.p.R_inv[2], ref.p.R_inv[3], ref.p.R_inv[4], ref.p.R_inv[5], ref.p.R_inv[6], ref.p.R_inv[7], ref.p.R_inv[8] };
	double refptinv[] = { ref.p.t_inv[0], ref.p.t_inv[1], ref.p.t_inv[2] };

	cudaMemcpyToSymbol(const_refpKinv, refpKinv, 9 * sizeof(double));
	cudaMemcpyToSymbol(const_refpRinv, refpRinv, 9 * sizeof(double));
	cudaMemcpyToSymbol(const_refptinv, refptinv, 3 * sizeof(double));

	int block_size_x = 32; // Nombre de threads par bloc dans la direction x
	int block_size_y = 32; // Nombre de threads par bloc dans la direction y
	dim3 block_dim(block_size_x, block_size_y);
	dim3 grid_dim((ref.width + block_size_x - 1) / block_size_x, (ref.height + block_size_y - 1) / block_size_y);

	// Taille de la mémoire partagée (extern __shared__)
	int shared_mem_size = (block_size_x + 2 * kernel_radius) * (block_size_y + 2 * kernel_radius) * sizeof(float);
	//dim3 block_size = dim3(Nx / thread_size.x, Ny / thread_size.y, 1); //(10,1080,1)
	for (auto& camera : cam_vector)
	{
		if (camera.name == ref.name)
			continue;

		//std::cout << "Cam: " << camera.name << std::endl;

		uint8_t* camera_yuv_0_arr = camera.YUV[0].isContinuous() ? camera.YUV[0].data : camera.YUV[0].clone().data;
		size_t length_camera = camera.YUV[0].total() * camera.YUV[0].elemSize();
		int width_camera = camera.YUV[0].cols;

		uint8_t* dev_camera_yuv_0_arr;
		CHK(cudaMalloc((void**)&dev_camera_yuv_0_arr, length_camera * sizeof(uint8_t)));
		CHK(cudaMemcpy(dev_camera_yuv_0_arr, camera_yuv_0_arr, length_camera * sizeof(uint8_t),
			cudaMemcpyHostToDevice));

		double campR[] = { camera.p.R[0], camera.p.R[1] , camera.p.R[2] , camera.p.R[3] , camera.p.R[4] , camera.p.R[5] , camera.p.R[6] , camera.p.R[7] , camera.p.R[8] };
		double campK[] = { camera.p.K[0], camera.p.K[1] , camera.p.K[2] , camera.p.K[3] , camera.p.K[4] , camera.p.K[5] };
		double campt[] = { camera.p.t[0], camera.p.t[1] , camera.p.t[2] };

		cudaMemcpyToSymbol(const_campR, campR, 9 * sizeof(double));
		cudaMemcpyToSymbol(const_campK, campK, 6 * sizeof(double));
		cudaMemcpyToSymbol(const_campt, campt, 3 * sizeof(double));

		cam* dev_ref, * dev_camera;



		CHK(cudaMalloc((void**)&dev_ref, sizeof(cam)));
		CHK(cudaMalloc((void**)&dev_camera, sizeof(cam)));

		CHK(cudaMemcpy(dev_ref, &ref, sizeof(cam),
			cudaMemcpyHostToDevice));

		CHK(cudaMemcpy(dev_camera, &camera, sizeof(cam),
			cudaMemcpyHostToDevice));


		//dim3 block_size = dim3(Nx / thread_size.x, Ny / thread_size.y, 1); //(10,1080,1) 
		compute_shared_constante << <grid_dim, block_dim, shared_mem_size >> > (dev_ref_yuv_0_arr, dev_camera_yuv_0_arr, wids, heig, dev_cost_cube);
		CHK(cudaMemcpy(cost_cube, dev_cost_cube, dev_cost_length,
			cudaMemcpyDeviceToHost));
		CHK(cudaDeviceSynchronize());

		cudaFree(dev_camera_yuv_0_arr);


	}
Error:
	cudaFree(dev_cost_cube);
	cudaFree(dev_ref_yuv_0_arr);




	return cost_cube;

}