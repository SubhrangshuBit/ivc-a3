### CS585 Image and Video Computing - Assignment 3

**Collaborators:** Nazia Tasnim and Subhrangshu Bit

#### Single Object Tracking with a Bayesian Recursive Filter
Given detected 2D locations of a vehicle over multiple frames in the video, you need to implement an alpha-beta filter or a Kalman filter to generate a smooth 2D track of the vehicle based on these 2D observations.

Related files:
```bash
├── part_1_alpha_beta_filter.mp4
├── part_1_alpha_beta.ipynb
├── part_1_demo.mp4
├── part_1_kalman_filter.mp4
├── part_1_kalman.ipynb
├── part_1_object_tracking_base.json
├── part_1_object_tracking.json
```
![part1](https://i.imgur.com/wUNMKOs.jpeg)
#### Multi-Object Tracking and Data Association
Given the bounding boxes of multiple objects detected in the video, you need to track them by assigning a unique ID to each object over the video as long as they are detected.

Related files:
```bash
├── part_2_demo.mp4
├── part_2_frame_dict.json
├── part_2_tracking_greedy.ipynb
├── part_2_assignment.ipynb
├── hungarian.py
```
![part2](https://i.imgur.com/4EIEFit.jpeg)
