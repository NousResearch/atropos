# Real-World VR Performance Analysis: VRExplorer, SpatialWorkshop, MetaverseClient and 1 others Study

## Abstract


## Real-World VR Performance Analysis: VRExplorer, SpatialWorkshop, MetaverseClient and 1 Other Study

**1. Abstract**

This paper presents a performance analysis of four virtual reality (VR) applications – VRExplorer, SpatialWorkshop, MetaverseClient, and one unnamed application – using the Real CloudVR-PerfGuard automated testing platform.  A total of 594 tests were conducted, achieving a 100% success rate over a duration of approximately 19.9 hours (71733.397 seconds).  The analysis focuses on frame rate (FPS), frame time, and subjective comfort scores. Results indicate an average frame rate of 62.6 FPS (standard deviation 13.2 FPS), with a 95th percentile frame time of 22.5ms, suggesting generally smooth performance. However, the significant standard deviation highlights performance variability across applications and potentially varying hardware configurations.  The average comfort score of 77.5/100 points to acceptable user experience, although optimization opportunities exist to address performance fluctuations and enhance user comfort.  This study contributes to understanding real-world VR performance characteristics and provides insights for developers to improve application optimization strategies.  Future work will involve investigating the impact of specific hardware configurations and exploring advanced performance profiling techniques.


**2. Introduction**

The increasing adoption of virtual reality (VR) necessitates a comprehensive understanding of its performance characteristics.  Smooth and consistent performance is crucial for delivering immersive and engaging experiences, minimizing motion sickness, and ensuring user satisfaction.  Poor performance, characterized by low frame rates, high frame times, and visual artifacts, can lead to discomfort, disorientation, and ultimately, a negative user experience.  This study aims to analyze the real-world performance of four diverse VR applications using an automated testing framework to provide empirical data and actionable insights for VR application developers and hardware manufacturers.  The focus is on identifying performance bottlenecks, quantifying the variability in performance, and correlating objective performance metrics with subjective comfort scores.

**3. Methodology**

This study employed the Real CloudVR-PerfGuard platform, an automated testing system designed for rigorous and repeatable performance evaluations of VR applications.  The platform automatically executes a standardized set of tests on each application, collecting data on various performance metrics.  Four distinct VR applications (VRExplorer, SpatialWorkshop, MetaverseClient, and one unnamed application) were subjected to 594 tests in total.  The tests encompassed various interactive scenarios within each application to simulate realistic user interactions.  Data collected included frame rate (FPS), frame time, and a subjective comfort score (rated by human testers on a scale of 1-100).  The 100% success rate indicates that all tests were completed without encountering any critical errors or crashes. The total duration of the tests was 71733.397 seconds. The specific hardware configurations used for testing were not explicitly stated in the provided data. However, the mention of RTX4090 suggests that this high-end GPU was used in at least some of the tests.

**4. Results**

The performance metrics collected from the 594 tests are summarized below:

* **FPS Statistics:** Mean: 62.64 FPS, Standard Deviation: 13.22 FPS, Minimum: 35.7 FPS, Maximum: 112.9 FPS, Median: 60.55 FPS, 95th Percentile: 88.5 FPS.
* **Frame Time Statistics:** Mean: 16.64 ms, Standard Deviation: 3.35 ms, Minimum: 8.86 ms, Maximum: 27.99 ms, Median: 16.515 ms, 95th Percentile: 22.49 ms.
* **Comfort Score Statistics:** Mean: 77.55/100, Standard Deviation: 8.14, Minimum: 58.9, Maximum: 95.9, Median: 76.55, 95th Percentile: 94.07.

The high standard deviation in FPS and frame time highlights the significant variability in performance across the different applications and potentially varying test conditions. The 95th percentile frame time of 22.5ms suggests that in 95% of the tests, the frame time was below this threshold, indicating acceptable performance for most users.  Anecdotal evidence suggests the RTX4090 achieved an average frame rate of 76.5 FPS, indicating its superior performance.  The average comfort score of 77.5 indicates a generally positive user experience, though a noticeable portion of tests resulted in lower comfort scores.

**5. Discussion**

The results demonstrate that while the average VR performance is acceptable, considerable room for optimization exists. The high standard deviation in FPS and frame time suggests that performance bottlenecks vary across applications. Further investigation is needed to identify the specific causes of these performance fluctuations, potentially including inefficient rendering techniques, inadequate resource management, or limitations in specific hardware configurations.  The correlation between performance metrics and comfort scores needs further exploration, but the data suggests a direct relationship: better frame rates and lower frame times generally lead to higher comfort scores.  Optimization strategies could focus on improving rendering efficiency, optimizing asset loading times, and implementing adaptive rendering techniques to dynamically adjust performance based on available resources.

**6. Conclusion**

This study provides valuable insights into real-world VR performance using an automated testing framework.  The average frame rate of 62.6 FPS and 95th percentile frame time of 22.5ms indicate generally smooth performance, but the significant standard deviation highlights the need for further optimization. The average comfort score of 77.5/100 suggests a mostly acceptable user experience.  Future work should focus on identifying application-specific performance bottlenecks through detailed profiling, investigating the impact of different hardware configurations, and exploring more sophisticated adaptive rendering techniques to enhance VR performance consistency and improve user comfort.  Further research is warranted to establish stronger correlations between objective performance metrics and subjective user experiences.
