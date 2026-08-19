/* v4_config.h — the shape of the batch-tiled kernel.
 *
 * Shared by v4_batch.cu and the CPU simulation in selection_sim.cpp, for the
 * same reason as v3_config.h: a simulation that does not run the kernel's own
 * decomposition proves nothing about the kernel.
 */
#ifndef KERNELFORGE_V4_CONFIG_H
#define KERNELFORGE_V4_CONFIG_H

#define V4_CHUNK     256  /* documents per block */
#define V4_QT          8  /* queries per block — the reuse factor for every X byte */
#define V4_BLOCK     256  /* threads per block (8 warps) */
#define V4_PER_LANE  (V4_CHUNK / 32)   /* scores each lane scans during selection */

#endif
