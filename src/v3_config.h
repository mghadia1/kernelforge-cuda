/* v3_config.h — the shape of v3's two-stage selection.
 *
 * v3_topk.cu and selection_sim.cpp both include this. The simulation is only
 * worth anything if it runs the same decomposition as the kernel, so the
 * constants that define that decomposition live in one place and neither file
 * is allowed its own copy.
 */
#ifndef KERNELFORGE_V3_CONFIG_H
#define KERNELFORGE_V3_CONFIG_H

#define KF_CHUNK       1024  /* documents scored per block */
#define KF_V3_BLOCK     256  /* threads per scoring block (8 warps) */
#define KF_MERGE_BLOCK  256  /* threads per merge block */
#define KF_MAX_K          8  /* caps the shared-memory reservation; k = 5 in practice */

#endif
