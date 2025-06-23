# Real-World VR Performance Analysis: SpatialWorkshop, VRTraining, MetaverseClient and 1 others Study

## Abstract


## Real-World VR Performance Analysis: SpatialWorkshop, VRTraining, MetaverseClient and Others

**1. Abstract**

This paper presents a performance analysis of four real-world Virtual Reality (VR) applications – SpatialWorkshop, VRTraining, MetaverseClient, and one unnamed application – utilizing a novel automated testing framework, Real CloudVR-PerfGuard.  A total of 614 tests were conducted, achieving a 100% success rate over a cumulative duration of 73947.25 seconds.  The study focused on key performance metrics including frames per second (FPS), frame time, and subjective user comfort scores.  Results reveal an average FPS of 62.0 ± 13.2, with a 95th percentile frame time of 22.6ms, indicating generally smooth performance.  However, the significant standard deviation in FPS highlights performance variability.  The average comfort score of 77.1/100 suggests room for improvement in certain applications or scenarios.  This research provides valuable insights into real-world VR performance characteristics and identifies optimization opportunities for developers to enhance both performance and user experience, specifically focusing on minimizing frame time variability and improving the comfort scores for specific application bottlenecks.  The RTX 4090 demonstrated superior performance, reaching an average FPS of 75.6, suggesting hardware optimization as a potential area for enhancing VR experiences.


**2. Introduction**

The growing adoption of Virtual Reality (VR) technologies across various sectors necessitates a comprehensive understanding of performance characteristics and their impact on user experience.  Poor VR performance, manifested as low frame rates (FPS), high frame times, and stuttering, can lead to motion sickness, visual discomfort, and ultimately, user dissatisfaction.  Maintaining a consistently smooth and immersive experience is crucial for maximizing the effectiveness of VR applications in fields such as training, design, and entertainment.  This study investigates the real-world performance of four diverse VR applications using a robust automated testing methodology to provide quantitative data and identify areas for optimization.  Understanding these performance characteristics is vital for developers striving to create high-quality, engaging, and accessible VR experiences.

**3. Methodology**

This research employed Real CloudVR-PerfGuard, an automated testing framework designed to objectively assess VR application performance. The framework executes pre-defined scenarios within each application, capturing performance metrics in real-time.  Four distinct VR applications (SpatialWorkshop, VRTraining, MetaverseClient, and one unnamed application) underwent rigorous testing.  Each application was subjected to a diverse range of scenarios designed to represent typical user interactions.  A total of 614 successful tests were conducted, resulting in a 100% success rate.  The total testing duration spanned 73947.25 seconds.  Performance metrics collected included FPS, frame time, and a subjective comfort score (rated on a scale of 1-100 by human testers for each run, where 100 represents optimal comfort).  The RTX 4090 GPU was identified as the benchmark high-end hardware.


**4. Results**

The collected data reveals the following key performance metrics:

* **FPS Statistics:**  The average FPS across all tests was 62.0 ± 13.2 (mean ± standard deviation).  The minimum FPS observed was 35.5, while the maximum reached 110.7. The median FPS was 59.9, and the 95th percentile was 87.8.  This indicates a substantial variation in performance across different applications and scenarios.

* **Frame Time Statistics:**  The average frame time was 16.82ms ± 3.38ms.  The minimum frame time was 9.03ms, and the maximum was 28.18ms. The median frame time was 16.69ms, and the 95th percentile was 22.6ms.  The relatively high standard deviation highlights the inconsistency in frame time.

* **Comfort Score Statistics:**  The average comfort score was 77.1 ± 8.1.  This suggests that while generally acceptable,  there is room for substantial improvement in user comfort levels for certain applications and scenarios. The minimum score was 58.7, and the maximum was 95.9. The median score was 76.15, with a 95th percentile of 93.67. The RTX 4090 GPU showcased superior performance with an average FPS of 75.6.

**5. Discussion**

The relatively high standard deviation in FPS and frame time indicates significant performance fluctuations within and across the tested applications. This variability likely contributes to user discomfort and reduced immersion.  The average comfort score of 77.1/100 suggests that while the overall performance is adequate, optimizing for consistency and reducing frame time variability is crucial for improving user experience.  The superior performance of the RTX 4090 GPU underscores the importance of hardware optimization, though software optimization remains critical for mitigating performance bottlenecks across various hardware configurations.  Further investigation is required to identify specific scenarios and application components contributing to performance fluctuations. This may involve profiling individual application modules to pinpoint bottlenecks.

**6. Conclusion**

This study provides a comprehensive analysis of the real-world performance of four diverse VR applications using automated testing.  The results highlight the importance of consistent performance and the impact of frame time variability on user comfort. While the average FPS of 62.0 is acceptable, the substantial standard deviation necessitates further investigation and optimization efforts.  Future work will focus on identifying specific performance bottlenecks within each application, correlating performance metrics with specific user interactions, and exploring optimization techniques to minimize frame time variability and improve overall VR comfort scores.  The utilization of advanced profiling tools and the development of more sophisticated automated testing methodologies are crucial for addressing these challenges and driving advancements in VR technology.
