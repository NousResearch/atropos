# Comparative VR Performance Analysis: SpatialWorkshop vs MetaverseClient vs VRTraining and 1 Other Applications

## Abstract


## Comparative VR Performance Analysis: SpatialWorkshop vs MetaverseClient vs VRTraining and Other Applications

**1. Abstract**

This paper presents a comparative performance analysis of four virtual reality (VR) applications – SpatialWorkshop, MetaverseClient, VRTraining, and one unnamed application – using the Real CloudVR-PerfGuard automated testing platform.  The study encompassed 594 test runs, achieving a 100% success rate across a total duration of approximately 19.9 hours.  Key performance metrics, including frames per second (FPS), frame time, and subjective comfort scores, were collected and statistically analyzed. Results indicate an average FPS of 62.6 (std: 13.2), with a 95th percentile frame time of 22.5ms, suggesting generally smooth performance.  The average comfort score of 77.5/100 highlights areas for potential optimization.  The RTX 4090 GPU demonstrated superior performance, achieving an average FPS of 76.5.  This research provides valuable insights into the performance characteristics of contemporary VR applications and identifies potential avenues for enhancing VR experience quality through targeted optimization strategies.  Future work will focus on investigating the impact of specific hardware configurations and application features on performance.


**2. Introduction**

The increasing prevalence of virtual reality (VR) applications across diverse sectors, including entertainment, training, and collaboration, necessitates rigorous performance evaluation.  Smooth and responsive VR experiences are crucial for user immersion, engagement, and preventing motion sickness.  Poor performance, characterized by low frame rates (FPS), high frame times, and visual artifacts, can significantly detract from the overall user experience and limit the potential of VR technology.  This study aims to provide a comprehensive performance analysis of four representative VR applications, leveraging an automated testing framework to provide a statistically robust assessment of their performance characteristics and identify potential areas for optimization.


**3. Methodology**

This research utilized the Real CloudVR-PerfGuard platform for automated performance testing.  This platform allows for standardized testing across various hardware configurations, eliminating inconsistencies associated with manual testing.  Four different VR applications – SpatialWorkshop, MetaverseClient, VRTraining, and an undisclosed application – were subjected to 594 independent test runs.  The platform automatically collected and recorded various performance metrics, including FPS, frame time, and a subjective comfort score (on a scale of 0-100, derived from user feedback integrated into the PerfGuard system).  The duration of the entire testing phase was 71733.397 seconds (approximately 19.9 hours).  All test runs were successfully completed, resulting in a 100% success rate.


**4. Results**

The collected performance data was statistically analyzed to obtain descriptive statistics.  The following key findings are reported:

* **FPS Statistics:** The mean FPS was 62.6 (std: 13.2), with a minimum of 35.7 FPS and a maximum of 112.9 FPS. The median FPS was 60.55, and the 95th percentile was 88.5 FPS.
* **Frame Time Statistics:** The mean frame time was 16.64ms (std: 3.35ms), with a minimum of 8.86ms and a maximum of 27.99ms. The median frame time was 16.515ms, and the 95th percentile was 22.49ms.
* **Comfort Score Statistics:**  The average comfort score was 77.5/100 (std: 8.14), indicating a generally acceptable level of comfort, although with room for improvement.  Minimum comfort score was 58.9 and maximum was 95.9.  The median score was 76.55, and the 95th percentile was 94.07.
* **GPU Performance:** Preliminary analysis indicated that the RTX 4090 GPU consistently outperformed other tested GPUs, achieving an average FPS of 76.5.  Further research is needed to assess GPU performance across a wider range of models.

**5. Discussion**

The results demonstrate generally good performance across the tested VR applications.  However, the standard deviation in FPS (13.2) indicates considerable variability, suggesting that performance may be significantly affected by factors such as scene complexity, application-specific rendering techniques, and hardware limitations.  The 95th percentile frame time of 22.5ms is relatively low, suggesting that the majority of frames are rendered within an acceptable timeframe for smooth VR experiences.  However, instances exceeding this threshold (represented by the maximum frame time of 27.99ms) could contribute to visual artifacts and discomfort.  The average comfort score of 77.5/100, while above average, suggests opportunities for optimization to enhance user experience and reduce motion sickness.  Further investigation is required to identify the specific application features or scene elements contributing to lower comfort scores.


**6. Conclusion**

This study provides a comprehensive performance evaluation of four diverse VR applications using an automated testing approach.  The results highlight the importance of considering both average and percentile-based performance metrics to provide a holistic understanding of VR application performance.  The identified areas for optimization, including reducing frame time variability and enhancing the overall comfort score, will be explored in future research. This includes investigating the effects of specific hardware and software configurations, optimization techniques, and user-specific factors on VR performance and user experience.  Future work will also focus on expanding the scope of applications tested and incorporating more detailed analysis of individual application performance characteristics.
