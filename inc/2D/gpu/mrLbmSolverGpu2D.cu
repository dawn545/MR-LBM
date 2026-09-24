
#include "../../../common/mlcudaCommon.h"
#include "mrConstantParamsGpu2D.h"
#include "mrUtilFuncGpu2D.h"
#include "mrLbmSolverGpu2D.h"


__global__ void mrSolver2DKernel(
	mrFlow2D* mlflow, /*float* vx_dev,*/ int sample_x, int sample_y, int sample_num)
{
	// ===== ASSIGNMENT FILL BEGIN: P1-C GPU LBM update kernel =====
	// TODO: Implement one complete pull-streaming and collision update.
	// 1. Map the CUDA thread to a lattice node and process ML_FLUID nodes only.
	// 2. Pull each D2Q9 population from its upstream neighbor by reconstructing
	//    it from the six stored moments.
	// 3. Recover density, force-corrected velocity, and raw second moments.
	// 4. Compute omega from viscosity, collide the second moments, normalize
	//    them, and write all six values to fMomPost.
	// Boundary and solid moments are prescribed by the case layer.

	// -------------------------------------------------------------------
	// 步骤 1：把 CUDA 线程映射到格点 (x, y)
	// grid/block 由 mrSolver2DGpu 按 BLOCK_NX x BLOCK_NY 划分，线程总数
	// 会向上取整超过网格，所以越界的线程必须直接返回。
	// -------------------------------------------------------------------
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= sample_x || y >= sample_y)
	{
		return;
	}

	// 只有 ML_FLUID 格点由本 kernel 推进；入口/出口/固壁/固体的矩由
	// CPU 边界层（RefreshDemoCaseBoundaries / ApplyObstaclePose）写好，
	// kernel 不能覆盖它们。
	const int index = y * sample_x + x;
	if (mlflow[0].flag[index] != ML_FLUID)
	{
		return;
	}

	// -------------------------------------------------------------------
	// 步骤 2：Pull streaming（拉取式迁移）
	// 对每个方向 i，粒子从上游邻居沿方向 i 迁移过来，所以上游坐标是
	// (x - ex[i], y - ey[i])。我们不长期存 9 个分布函数，而是从邻居存的
	// 6 个矩即时重构出 f_i。
	// -------------------------------------------------------------------
	mrUtilFuncGpu2D util;
	REAL f[9];
	for (int i = 0; i < 9; i++)
	{
		// 上游格点坐标；越界时 clamp 回边界格点（其 ghost 矩已由 CPU 写好，
		// 相当于把边界值当作来流，保证近壁/近边界方向也有合理的 f_i）。
		int nx = x - (int)ex2d_gpu[i];
		int ny = y - (int)ey2d_gpu[i];
		if (nx < 0) nx = 0;
		if (nx >= sample_x) nx = sample_x - 1;
		if (ny < 0) ny = 0;
		if (ny >= sample_y) ny = sample_y - 1;
		const int neighborIdx = ny * sample_x + nx;
		const REAL* neighborMom = &mlflow[0].fMom[neighborIdx * 6];

		// 注意参数顺序：函数签名是 (rho, ux, uy, pixx, pixy, piyy, ...)，
		// 而存储顺序是 [3]=pixx [4]=piyy [5]=pixy，所以这里必须传
		// neighborMom[3], neighborMom[5], neighborMom[4]（pixx, pixy, piyy）。
		// 这是最容易写反的地方。
		util.mlCalDistributionD2Q9AtIndex(
			neighborMom[0], neighborMom[1], neighborMom[2],
			neighborMom[3], neighborMom[5], neighborMom[4],
			i, f[i]);
	}

	// -------------------------------------------------------------------
	// 步骤 3：由重构出的 9 个 f_i 恢复宏观量
	//   rho   = Σ f_i                          （零阶矩 / 密度）
	//   Σ ex*f, Σ ey*f                          （一阶矩 / 动量）
	//   Σ ex*ex*f, Σ ey*ey*f, Σ ex*ey*f         （原始二阶矩）
	// -------------------------------------------------------------------
	REAL rho = 0.0f;
	REAL ux_times_rho = 0.0f;
	REAL uy_times_rho = 0.0f;
	REAL pixx_raw = 0.0f;
	REAL piyy_raw = 0.0f;
	REAL pixy_raw = 0.0f;
	for (int i = 0; i < 9; i++)
	{
		rho += f[i];
		ux_times_rho += f[i] * ex2d_gpu[i];
		uy_times_rho += f[i] * ey2d_gpu[i];
		pixx_raw += f[i] * ex2d_gpu[i] * ex2d_gpu[i];
		piyy_raw += f[i] * ey2d_gpu[i] * ey2d_gpu[i];
		pixy_raw += f[i] * ex2d_gpu[i] * ey2d_gpu[i];
	}

	// 体力半步修正：速度需加上 0.5*F/rho（Guo 力格式的一半在此处、
	// 另一半在碰撞里）。本项目 gy=0，forcex/forcey 通常为 0，此项为 0，
	// 但保留以支持带体力的推广。
	const REAL Fx = mlflow[0].forcex[index];
	const REAL Fy = mlflow[0].forcey[index];
	const REAL ux = (ux_times_rho + 0.5f * Fx) / rho;
	const REAL uy = (uy_times_rho + 0.5f * Fy) / rho;

	// -------------------------------------------------------------------
	// 步骤 4：二阶矩碰撞（BGK 松弛 + 体力修正）
	// 松弛时间 tau 与运动黏度关系：nu = cs2 * (tau - 0.5)，
	// 反解得 tau = nu/cs2 + 0.5 = 3*nu + 0.5，omega = 1/tau。
	// mlGetPIAfterCollision 原地把 pixx_raw/piyy_raw/pixy_raw 松弛到平衡态。
	// -------------------------------------------------------------------
	const REAL vis = mlflow[0].vis_shear;
	const REAL tau = vis / cs2 + 0.5f;
	const REAL omega = 1.0f / tau;
	util.mlGetPIAfterCollision(
		rho, ux, uy, Fx, Fy, omega, pixx_raw, piyy_raw, pixy_raw);

	// -------------------------------------------------------------------
	// 步骤 5：归一化并写回 fMomPost（下一时刻缓冲）
	// 归一化定义 pi = raw/rho - cs2（切应力用），必须与 CPU 端
	// WriteEquilibriumMoments2D 完全一致，否则边界与内部不匹配会发散。
	// 写入 fMomPost 而非 fMom：双缓冲交换由 mrSolver2D_step2Kernel 里的
	// MomSwap 完成。
	// -------------------------------------------------------------------
	REAL* post = &mlflow[0].fMomPost[index * 6];
	post[0] = rho;
	post[1] = ux;
	post[2] = uy;
	post[3] = pixx_raw / rho - cs2;
	post[4] = piyy_raw / rho - cs2;
	post[5] = pixy_raw / rho;
	return;
	// ===== ASSIGNMENT FILL END: P1-C =====
}
__host__ __device__
void MomSwap(REAL*& pt1, REAL*& pt2) {
	REAL* temp = pt1;
	pt1 = pt2;
	pt2 = temp;
}
__global__ void mrSolver2D_step2Kernel(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	MomSwap(mlflow[0].fMom, mlflow[0].fMomPost);
}


void mrSolver2DGpu(mrFlow2D* mlflow, MLFluidParam2D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_num = sample_x * sample_y;

	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y), 1
	);
	mrSolver2DKernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	mrSolver2D_step2Kernel << <1, 1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}
