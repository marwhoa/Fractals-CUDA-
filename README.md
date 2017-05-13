cudaJulia.cu was designed in such a way as to not incorporate shared memory or constant memory.
Constant memory is utilized when you have data that does not change during kernel execution, and all of the threads in the data can access it.
I was going to make the struct data in dd_data and ii_data Constant Memory, but I realized that the values bundled in these arrays actually changed based on
the threadIdx.x and threadIdx.y values that were running it. For this same specific reason, Shared Memory was not used for these values.

There are a lot of variables to be testing that in general could affect the timing of the code. I was thinking that if I increaed the number of maxIterations,
or the complexity of the fractal generation, the CUDA code would increasingly get more efficient than the serial code. I ran a few tests,
and found out this wasn't completely true. In general, it appeared that the performance increase of the CUDA code vs. serial code was around 325-350 times faster,
regardless of what parameters changed.

For example, in the most basic case, I started with the default image, performed a series of image moves and zooms, and recorded the images.

Serial Time:
Zoom in by 5000: 20.92 seconds
Move Down: 27.6 seconds
Zoom in by 2500: 27.17 seconds
Move up: 27.98 seconds

Parallel Time:
Zoom in by 5000: .060564 seconds
Move Down: .062960 seconds
Zoom in by 2500: .079666 seconds
Move up: .077447

The most significant change that affected timing was increasing the number of iterations per image generation.
Of course, this change scaled lineraly, so the performance increase from host to parallel code was the same.

-Aaron
