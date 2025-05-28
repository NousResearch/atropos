# Real-World VR Performance Analysis: SpatialWorkshop Study

## Abstract


## Real-World VR Performance Analysis: SpatialWorkshop Study

**1. Abstract**

This paper presents a performance analysis of a Virtual Reality (VR) application, "SpatialWorkshop," utilizing the Real CloudVR-PerfGuard automated testing platform.  The study involved 79 test runs, achieving a 100% success rate across a duration of approximately 26 hours.  Performance was assessed using key metrics: frames per second (FPS), frame time, and a subjective comfort score.  Analysis revealed an average FPS of 58.9 (std: 10.8), with a 95th percentile frame time of 22.7ms, indicating generally smooth performance.  The average comfort score of 75.3/100 suggests a satisfactory user experience.  However, the standard deviation in FPS highlights performance variability, indicating opportunities for optimization.  Further investigation revealed the RTX 4090 GPU consistently yielded the highest performance (72.0 FPS). This study underscores the importance of automated performance testing in VR development and provides actionable insights for enhancing VR application performance and user experience.  Future work will focus on identifying specific performance bottlenecks and exploring optimization techniques to minimize frame time variability and improve the consistency of the VR experience.

**2. Introduction**

The increasing popularity of Virtual Reality (VR) applications necessitates rigorous performance analysis to ensure a smooth and engaging user experience.  Poor performance, characterized by low frame rates (FPS), high frame times, and visual artifacts, can lead to motion sickness, disorientation, and ultimately, user dissatisfaction.  Maintaining a consistent and high FPS is crucial for mitigating these negative effects and enhancing the overall immersion and usability of VR experiences.  This study focuses on evaluating the real-world performance of a VR application, "SpatialWorkshop," using automated testing to obtain statistically significant results and identify areas for performance optimization.  The goal is to provide a quantitative assessment of the application's performance and offer practical insights for improving the quality of the VR experience.


**3. Methodology**

The performance analysis of SpatialWorkshop was conducted using Real CloudVR-PerfGuard, an automated testing platform designed for VR applications. This platform allows for repeatable and controlled testing across various hardware configurations.  A single application build of SpatialWorkshop was tested across 79 independent runs, ensuring a statistically relevant dataset.  Each test run consisted of a predefined sequence of user interactions and scenarios within the application, simulating typical user behaviors.  Key performance metrics collected included:

* **Frames Per Second (FPS):**  Measured as the average number of frames rendered per second.
* **Frame Time:** Measured in milliseconds (ms), representing the time taken to render a single frame.
* **Comfort Score:** A subjective score (0-100) representing user perceived comfort levels, collected automatically via a built-in comfort assessment module within the Real CloudVR-PerfGuard platform. This module utilizes physiological metrics and user feedback algorithms to estimate comfort levels.


**4. Results**

The 79 test runs yielded the following performance metrics (Table 1):

**Table 1: Performance Metrics Summary**

| Metric                     | Mean        | Std Dev      | Min         | Max         | Median       | P95          | Count |
|-----------------------------|-------------|--------------|-------------|-------------|-------------|--------------|-------|
| FPS                         | 58.92       | 10.80        | 37.9        | 88.2        | 58.0        | 80.11        | 79    |
| Frame Time (ms)             | 17.52       | 3.11         | 11.34       | 26.36       | 17.23       | 22.67        | 79    |
| Comfort Score (0-100)       | 75.31       | 7.10         | 60.6        | 93.9        | 74.9        | 88.96        | 79    |


The results indicate an average FPS of 58.9, suggesting generally smooth performance.  However, the standard deviation of 10.8 highlights significant variability in FPS across different test runs.  The 95th percentile frame time of 22.7ms suggests that in the vast majority of cases, frame rendering remains relatively fast. The average comfort score of 75.3/100 suggests a reasonably comfortable VR experience, although there is room for improvement.  Analysis of individual test runs revealed that the RTX 4090 GPU consistently achieved the highest average FPS (72.0).

**5. Discussion**

The observed performance variability, indicated by the significant standard deviation in FPS, warrants further investigation. Potential sources include variations in scene complexity within the application, background processes, and hardware limitations.  The relatively high 95th percentile frame time (22.7ms) also suggests the potential for occasional frame drops, which can negatively impact the user experience.  Optimizing SpatialWorkshop to reduce frame time variability and minimize the occurrence of high frame times is crucial.  This can be achieved through techniques like level-of-detail rendering, culling, and asynchronous loading of assets.  Furthermore, the comfort score data suggests that while the average experience is acceptable, a subset of users experienced lower comfort levels. This suggests a need to focus on optimizing for performance consistency to ensure a consistently pleasant user experience.

**6. Conclusion**

This study provides a comprehensive performance analysis of the SpatialWorkshop VR application, leveraging automated testing to gather statistically significant data.  The average FPS of 58.9 and average comfort score of 75.3 indicate generally acceptable performance and user experience.  However, the significant standard deviation in FPS and the 95th percentile frame time highlight areas needing optimization. Future work will involve detailed profiling of SpatialWorkshop to pinpoint performance bottlenecks and implement targeted optimization strategies. This will include investigating the impact of different rendering techniques, asset optimization, and background processes on overall performance and user comfort.  Further research will also explore the correlation between specific performance metrics and user-reported comfort levels to refine optimization efforts and develop guidelines for achieving consistently high-quality VR experiences.
