Disclaimer: The code was ran on 3050ti laptop GPU (which is relatively tiny) so the perf isn't as good as the GPUs dedicated in the original class :(


### First attempt

I tried to pass correctness while changing the code minimally.
The most obvious change I can make was to iterate the for loop on host and execute the pixel coloring for each bbox for each circle. This way, we don't need to change shadePixel and reuse code from refRenderer.cpp.

pseudo code version:
```
Clear image
for each circle
    update position and velocity
// in Host(CPU)
for each circle
    compute screen bounding box
    // send info to device(GPU) -- each thread handles pixels in bounding box
    for each pixels in bounding box
        compute pixel center point
        if center point is within the circle
            compute color of circle at point
            blend contribution of circle into image for this pixel
```

```text
❯ ./checker.py

Running scene: rgb...
[rgb] Correctness passed!
[rgb] Student times:  [2.0574, 2.5079, 2.5733]
[rgb] Reference times:  [2.7237, 2.5331, 2.2006]

Running scene: rand10k...
[rand10k] Correctness passed!
[rand10k] Student times:  [311.7718, 300.7261, 304.5479]
[rand10k] Reference times:  [8.5178, 11.7104, 8.6562]

Running scene: rand100k...
[rand100k] Correctness passed!
[rand100k] Student times:  [3913.2483, 3805.5947, 4632.0454]
[rand100k] Reference times:  [57.4361, 48.8941, 46.3839]

Running scene: pattern...
[pattern] Correctness passed!
[pattern] Student times:  [33.2512, 18.8193, 34.2197]
[pattern] Reference times:  [2.7154, 2.4991, 2.6428]

Running scene: snowsingle...
[snowsingle] Correctness passed!
[snowsingle] Student times:  [614.3751, 604.3537, 622.8539]
[snowsingle] Reference times:  [39.6018, 37.2674, 37.2422]

Running scene: biglittle...
[biglittle] Correctness passed!
[biglittle] Student times:  [861.9706, 888.6677, 878.2177]
[biglittle] Reference times:  [27.5874, 36.0399, 35.6442]

Running scene: rand1M...
[rand1M] Correctness passed!
[rand1M] Student times:  [72926.3339, 70844.075, 68580.7382]
[rand1M] Reference times:  [372.4539, 374.3449, 372.1648]

Running scene: micro2M...
[micro2M] Correctness passed!
  2. Build bbox from host data.[micro2M] Student times:  [136657.386, 135271.2727, 103632.734]
[micro2M] Reference times:  [667.9026, 675.5625, 685.6775]
------------
Score table:
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 2.2006           | 2.0574          | 9               |
| rand10k         | 8.5178           | 300.7261        | 2               |
| rand100k        | 46.3839          | 3805.5947       | 2               |
| pattern         | 2.4991           | 18.8193         | 3               |
| snowsingle      | 37.2422          | 604.3537        | 2               |
| biglittle       | 27.5874          | 861.9706        | 2               |
| rand1M          | 372.1648         | 68580.7382      | 2               |
| micro2M         | 667.9026         | 103632.734      | 2               |
--------------------------------------------------------------------------
|                                    | Total score:    | 24/72           |
--------------------------------------------------------------------------
```
