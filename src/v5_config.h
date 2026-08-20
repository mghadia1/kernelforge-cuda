/* v5_config.h — the shape of the register-blocked kernel.
 *
 * V5_CHUNK and the 32-lane selection geometry deliberately match v4's, so the
 * host-side simulation of v4's selection covers v5's unchanged. Only the
 * scoring loop differs.
 */
#ifndef KERNELFORGE_V5_CONFIG_H
#define KERNELFORGE_V5_CONFIG_H

#define V5_CHUNK     256  /* documents per block (same as v4) */
#define V5_QT          8  /* queries per block */
#define V5_DR          4  /* documents per warp held in registers at once */
#define V5_BLOCK     256  /* threads per block (8 warps) */
#define V5_PER_LANE  (V5_CHUNK / 32)

#endif
