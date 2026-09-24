#pragma once
#ifndef _MRUTILFUNCGU2DH_
#define _MRUTILFUNCGU2DH_

#include "cuda_runtime.h"
//#include "../../../lw_core_win/mlCoreWinHeader.h"
#include "../../../common/mlCoreWin.h"
#include "../../../common/mlLatticeNode.h"

#include "mrConstantParamsGpu2D.h"
class mrUtilFuncGpu2D
{
public:
	MLFUNC_TYPE void mlCalDistributionD2Q9AtIndex(
		REAL rho, REAL ux, REAL uy, REAL pixx, REAL pixy, REAL piyy, int i, REAL& f_out);
	MLFUNC_TYPE void mlGetPIAfterCollision(REAL R, REAL U, REAL V, REAL Fx, REAL Fy, REAL omega,
		REAL& pixx,
		REAL& piyy,
		REAL& pixy);
 

private:

};

inline MLFUNC_TYPE void mrUtilFuncGpu2D::mlCalDistributionD2Q9AtIndex(
	REAL rho, REAL ux, REAL uy, REAL pixx, REAL pixy, REAL piyy, int i, REAL& f_out)
{
	// ===== ASSIGNMENT FILL BEGIN: P1-A D2Q9 reconstruction =====
	// TODO: Reconstruct population f_i from density, velocity, and
	// normalized second moments. Build the required moment/Hermite
	// coefficients, then evaluate the polynomial for direction i with the
	// corresponding D2Q9 weight.
	#ifdef __CUDA_ARCH__
	const REAL a0 = rho;// 0阶矩
	const REAL ax = rho * ux;// 1阶矩
	const REAL ay = rho * uy;// 1阶矩
	const REAL axx = rho * pixx;// 2阶矩
	const REAL ayy = rho * piyy;// 2阶矩
	const REAL axy = rho * pixy;// 2阶矩
	const REAL axxy =
		-2.0f * rho * uy * ux * ux + 2.0f * axy * ux + axx * uy;// 3阶矩
	const REAL axyy =
		-2.0f * rho * ux * uy * uy + 2.0f * axy * uy + ayy * ux;// 3阶矩

	REAL polynomial = 0.0f;
	switch (i)
	{
	case 0:
		polynomial = a0 - 1.5f * axx - 1.5f * ayy;
		break;
	case 1:
		polynomial = a0 + 3.0f * ax + 3.0f * axx - 4.5f * axyy - 1.5f * ayy;
		break;
	case 2:
		polynomial = a0 - 1.5f * axx - 4.5f * axxy + 3.0f * ay + 3.0f * ayy;
		break;
	case 3:
		polynomial = a0 - 3.0f * ax + 3.0f * axx + 4.5f * axyy - 1.5f * ayy;
		break;
	case 4:
		polynomial = a0 - 1.5f * axx + 4.5f * axxy - 3.0f * ay + 3.0f * ayy;
		break;
	case 5:
		polynomial = a0 + 3.0f * ax + 3.0f * axx + 9.0f * axy +
			9.0f * axxy + 9.0f * axyy + 3.0f * ay + 3.0f * ayy;
		break;
	case 6:
		polynomial = a0 - 3.0f * ax + 3.0f * axx - 9.0f * axy +
			9.0f * axxy - 9.0f * axyy + 3.0f * ay + 3.0f * ayy;
		break;
	case 7:
		polynomial = a0 - 3.0f * ax + 3.0f * axx + 9.0f * axy -
			9.0f * axxy - 9.0f * axyy - 3.0f * ay + 3.0f * ayy;
		break;
	case 8:
		polynomial = a0 + 3.0f * ax + 3.0f * axx - 9.0f * axy -
			9.0f * axxy + 9.0f * axyy - 3.0f * ay + 3.0f * ayy;
		break;
	}

	f_out = w2d_gpu[i] * polynomial;
	if (f_out < 0.0f)
	{
		f_out = 0.0f;
	}
#else
	f_out = 0.0f;
#endif

	return;

	// ===== ASSIGNMENT FILL END: P1-A =====
}

inline MLFUNC_TYPE void mrUtilFuncGpu2D::mlGetPIAfterCollision(REAL R, REAL U, REAL V, REAL Fx, REAL Fy, REAL omega, REAL& pixx, REAL& piyy, REAL& pixy)
{
	// ===== ASSIGNMENT FILL BEGIN: P1-B second-moment collision =====
	// TODO: Apply the BGK relaxation and body-force correction to the
	// three raw second moments in place. The equilibrium contribution must be
	// consistent with cs^2 = 1/3 and the supplied macroscopic velocity.
	#ifdef __CUDA_ARCH__
// =========================================================================
// HOME-LBM (NOCM-MRT) 二阶中心矩碰撞与外力项修正
// =========================================================================

// 0. 定义松弛率 (MRT 核心：分离剪切与体积松弛)
// 如果你的框架中尚未独立定义体积松弛率 (omega_bulk)，可以先将其设为 1.0f 或等于 omega_s
const REAL omega_s = omega;          // 剪切松弛率 (对应运动粘度)
const REAL omega_b = 1.0f;           // 体积松弛率 (控制体积粘度，通常取 1.0-1.2 之间提升稳定性)
// const REAL omega_b = omega;       // 如果不需要额外的体积粘度耗散，可回退为与 omega 相等

// 1. 将原始二阶矩转换为非正交中心矩 (Non-Orthogonal Central Moments)
// 物理意义：剥离宏观对流速度的影响，在随波逐流的参考系下进行碰撞
REAL sxx = pixx - R * U * U;
REAL syy = piyy - R * V * V;
REAL sxy = pixy - R * U * V;

// 2. 计算中心矩的平衡态 (等温模型的理论解)
const REAL sxx_eq = R * cs2;
const REAL syy_eq = R * cs2;
const REAL sxy_eq = 0.0f; // 剪切分量中心矩平衡态恒为 0

// 3. MRT 耦合松弛 (分离 Trace 和 Normal Stress)
// 对应论文中对 (Sxx + Syy) 和 (Sxx - Syy) 的解耦操作
REAL T = sxx + syy;             // 迹 (Trace)，对应体积形变
REAL T_eq = sxx_eq + syy_eq;    // = 2 * R * cs2

REAL N = sxx - syy;             // 法向应力差 (Normal stress difference)，对应剪切形变
REAL N_eq = 0.0f;               // = sxx_eq - syy_eq = 0

// 在中心矩空间独立松弛
T   -= omega_b * (T - T_eq);
N   -= omega_s * (N - N_eq);
sxy -= omega_s * (sxy - sxy_eq);

// 重构碰撞后的中心矩
sxx = 0.5f * (T + N);
syy = 0.5f * (T - N);

// 将碰撞后的中心矩转换回原始矩空间
pixx = sxx + R * U * U;
piyy = syy + R * V * V;
pixy = sxy + R * U * V;

// 4. MRT 兼容的外力项修正 (Guo Force in MRT)
// 外力对二阶原始矩的基础贡献源项
const REAL C_xx = 2.0f * U * Fx;
const REAL C_yy = 2.0f * V * Fy;
const REAL C_xy = U * Fy + V * Fx;

// 对应的力修正权重系数
const REAL forceFactor_b = 1.0f - 0.5f * omega_b;
const REAL forceFactor_s = 1.0f - 0.5f * omega_s;

// 将力源项同样按 Trace 和 Normal 模式拆分，匹配对应的权重
const REAL C_T = C_xx + C_yy;
const REAL C_N = C_xx - C_yy;

// 最终将外力修正叠加回原始矩
pixx += 0.5f * (forceFactor_b * C_T + forceFactor_s * C_N);
piyy += 0.5f * (forceFactor_b * C_T - forceFactor_s * C_N);
pixy += forceFactor_s * C_xy;
    #endif
	return;
	// ===== ASSIGNMENT FILL END: P1-B =====
}

 

#endif // !_MRUTILFUNCGU2DH_
