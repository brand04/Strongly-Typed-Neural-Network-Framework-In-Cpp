#pragma once


namespace NN {
	namespace CudaHelpers {
		static constexpr size_t MAX_BLOCK_SIZE = 256; //the maximum allowed block size
		static constexpr size_t MIN_BLOCK_SIZE = 64; //The minimum allowed block size
		static constexpr size_t DESIRED_GRID_SIZE = 20; // attempts to reach this grid size (or 1 above to handle excess threads), unless restricted by the other options
		static constexpr size_t WARP_SIZE = 32; //aligns the blocks size to this value

		static consteval size_t alignBlockSizeToWarp(size_t blockSize) {
			return (blockSize / WARP_SIZE) * WARP_SIZE;
		}

		//compute a reasonable block size

		/// <summary>
		/// Computes at compile time a reasonable thread size given the required number of threads
		/// </summary>
		/// <param name="requiredThreads">The total required number of threads</param>
		/// <returns>A block size</returns>
		static consteval size_t computeBlockSize(const size_t requiredThreads) {
			size_t desiredBlockSize = alignBlockSizeToWarp(requiredThreads / DESIRED_GRID_SIZE); //aligned to WARP_SIZE and producing DESIRED_GRID_SIZE grids
			if (desiredBlockSize <= MAX_BLOCK_SIZE && desiredBlockSize >= MIN_BLOCK_SIZE) {
				return desiredBlockSize;
			}
			else if (desiredBlockSize > MAX_BLOCK_SIZE) return alignBlockSizeToWarp(MAX_BLOCK_SIZE);
			else return alignBlockSizeToWarp(MIN_BLOCK_SIZE+(WARP_SIZE-1)); //increase so that it will round up if not equal to MIN_BLOCK_SIZE
		}

		//computes a good grid size. if blockSize==0, uses computeBlockSize(requiredThreads) as the blockSize instead

		/// <summary>
		/// Computes at compile time a reasonable grid size given the required number of threads and the block size used
		/// </summary>
		/// <param name="requiredThreads">The total required number of threads</param>
		/// <param name="blockSize">The block size being used, or if not provided (or set to 0), uses computeBlockSize(requiredThreads) to deduce the blockSize</param>
		/// <returns></returns>
		static consteval size_t computeGridSize(const size_t requiredThreads,  size_t blockSize = 0) {
			if  (blockSize == 0) blockSize = computeBlockSize(requiredThreads);
			return (requiredThreads / blockSize) + 1;
		}
	}
}