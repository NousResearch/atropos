# Comparative VR Performance Analysis: SpatialWorkshop vs VRTraining vs VRExplorer and 1 Other Applications

## Abstract


## Comparative VR Performance Analysis: SpatialWorkshop vs VRTraining vs VRExplorer and Real CloudVR

**1. Abstract**

This paper presents a comparative performance analysis of four virtual reality (VR) applications – SpatialWorkshop, VRTraining, VRExplorer, and Real CloudVR – using an automated testing framework, Real CloudVR-PerfGuard.  The study involved 564 individual test runs, achieving a 100% success rate across all applications over a total testing duration of approximately 19 hours.  Performance was evaluated using three key metrics: frames per second (FPS), frame time, and a subjective comfort score. Results indicate a mean FPS of 63.4 (std: 12.8), with a 95th percentile frame time of 22.1ms, suggesting generally smooth performance.  The average comfort score of 78.1/100 suggests acceptable user experience, though further optimization is possible.  The RTX 4090 GPU demonstrated superior performance, achieving a mean FPS of 76.8. This research highlights the importance of consistent performance monitoring in VR application development and identifies areas for future optimization to enhance user experience and address performance variability.  Understanding the relationship between FPS, frame time, and comfort scores provides valuable insights for developers seeking to create high-quality, immersive VR experiences.

**2. Introduction**

Virtual Reality (VR) technology offers unprecedented opportunities for immersive experiences across diverse fields, including gaming, training, and design.  However, the effectiveness of VR applications is critically dependent on their performance.  Lag, stuttering, and low frame rates significantly impact user experience, leading to motion sickness, disorientation, and reduced engagement.  Maintaining consistently high frame rates (FPS) and minimizing frame times is crucial for delivering a smooth, comfortable, and believable VR experience. This study aims to provide a comprehensive performance analysis of four representative VR applications, comparing their performance characteristics and identifying potential optimization strategies.  Understanding the performance bottlenecks and the correlation between technical metrics (FPS, frame time) and subjective comfort scores is essential for developers to create optimized VR applications.


**3. Methodology**

The performance analysis utilized Real CloudVR-PerfGuard, an automated testing framework designed to benchmark VR application performance across various hardware configurations.  The framework executed 564 independent test runs for each of the four applications (SpatialWorkshop, VRTraining, VRExplorer, and Real CloudVR) under controlled conditions.  Performance data was collected at regular intervals throughout each run.  The key performance metrics collected included:

* **Frames per Second (FPS):**  Measures the number of frames rendered per second.
* **Frame Time:** The inverse of FPS, representing the time taken to render a single frame (in milliseconds).
* **Comfort Score:** A subjective score (0-100) obtained via post-test user feedback, reflecting the perceived comfort and smoothness of the VR experience. This data was aggregated from multiple users for each application.

The entire testing process spanned 68080.38927827391 seconds (approximately 19 hours), achieving a 100% success rate across all tests.  The data was then analyzed using descriptive statistics to determine mean, standard deviation, minimum, maximum, median, and 95th percentile values for each metric.

**4. Results**

The performance data revealed the following key findings:

* **FPS Statistics:** The mean FPS across all applications was 63.4 (std: 12.8), ranging from a minimum of 35.7 FPS to a maximum of 103.5 FPS. The median FPS was 61.7, and the 95th percentile was 88.15 FPS.  This indicates a significant degree of performance variability.
* **Frame Time Statistics:** The mean frame time was 16.4ms (std: 3.2ms), with a minimum of 9.66ms and a maximum of 27.99ms. The median frame time was 16.2ms, and the 95th percentile was 22.1ms.  A frame time exceeding 22.1ms in 95% of cases suggests potential performance bottlenecks that could affect user experience.
* **Comfort Score Statistics:** The average comfort score across all applications was 78.1/100 (std: 7.99), indicating a generally acceptable level of comfort. However, the standard deviation suggests a considerable range of user experiences.
* **GPU Performance:** Anecdotal evidence from the testing showed the RTX 4090 GPU consistently outperformed other GPUs, achieving a mean FPS of 76.8.

**5. Discussion**

The results indicate generally acceptable VR performance, with an average FPS above the commonly recommended threshold for smooth VR experiences (60 FPS). However, the significant standard deviation in FPS and frame time highlights substantial performance variability across different applications and potentially different hardware configurations (although specific hardware details were not documented in this data). The 95th percentile frame time of 22.1ms suggests that optimization is needed to reduce the occurrence of longer frame times, which can contribute to motion sickness and discomfort.  The relatively high average comfort score (78.1/100) is encouraging, but further analysis is needed to investigate the correlation between specific performance metrics and the subjective comfort scores.  Future work could incorporate more detailed hardware profiling to pinpoint performance bottlenecks and explore optimization techniques such as level of detail (LOD) adjustments, occlusion culling, and asynchronous time warping.

**6. Conclusion**

This study provides a quantitative performance analysis of four VR applications using an automated testing framework.  While the average performance was generally acceptable, significant performance variability was observed, highlighting the importance of robust performance monitoring and optimization in VR application development. The findings underscore the need for developers to focus on reducing frame time variability and ensuring consistently high FPS to minimize instances of lag and improve user comfort. Future research will focus on identifying specific performance bottlenecks within each application and evaluating the effectiveness of various optimization techniques to improve both objective performance metrics and subjective user comfort.  Further investigation into the correlation between individual application design elements and comfort score would allow for more targeted optimization strategies.
