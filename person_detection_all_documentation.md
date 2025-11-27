# Person Detection

## Project Description
A person detection in school campus using CCTV camera, in a single block of school buildings, there have multiple camera, a single camera will detect and track the person that it detect and if a person appear in two or multiple camera scene, it will be detect as same and one person, the model in this project such as detect model and tracking model, will be flexible to change to other model

## Product Requirements Document
PRODUCT REQUIREMENTS DOCUMENT (PRD) - PERSON DETECTION SYSTEM

1. INTRODUCTION

1.1 Purpose
This Product Requirements Document (PRD) specifies the requirements for the Person Detection System to be deployed within a single block of school buildings utilizing existing CCTV infrastructure. This system will perform real-time person detection, single-camera tracking, and critically, Multi-Camera Re-Identification (Re-ID) to ensure consistent identity assignment across multiple camera feeds. This module serves as a prerequisite filter for a subsequent Cloth Detection module.

1.2 Goals
The primary goal is to reliably detect, track, and maintain a consistent global identity for every person observed across the specified camera network within the school block. The system must be flexible enough to allow for swapping underlying detection, tracking, and Re-ID models without significant architectural overhaul.

1.3 Scope
Initial deployment scope covers approximately 4 CCTV camera sources within a single, contiguous block of school buildings. The system must output standardized detection metadata for downstream processing (e.g., Cloth Detection).

2. TARGET USERS AND USE CASES

2.1 Target Users
Primary Users: School Administrators.
Role: Operate the system interface, monitor detection and tracking results, and use the system's output (filtered person crops) for the subsequent Cloth Detection module.

2.2 Use Cases
UC-01: Real-time detection and bounding box visualization across all active cameras.
UC-02: Consistent assignment of a Global ID to a person appearing sequentially across multiple cameras (Multi-Camera Re-ID).
UC-03: Providing standardized metadata (including Global ID and cropped image) of the detected person to the downstream Cloth Detection module.
UC-04: System operators monitor the overall health and visual output (bounding boxes) of the detection and tracking processes via the frontend interface.

3. FUNCTIONAL REQUIREMENTS

3.1 Detection and Tracking (Single Camera)
FR-01: The system shall detect persons in each video frame from the designated camera sources.
FR-02: The system shall assign a unique local track ID to each detected person within a single camera feed.
FR-03: The system must process video streams concurrently for all active cameras (initially 4 sources).

3.2 Multi-Camera Re-Identification (Re-ID)
FR-04: The system shall implement a Re-ID module capable of comparing features of a track from one camera against existing tracks/persons from all other cameras.
FR-05: The Re-ID mechanism must successfully match the same individual across different camera views, accounting for significant variations in:
    - Body pose and gait.
    - Lighting and exposure differences.
    - Camera angle and viewpoint change.
    - Partial occlusion or crowding.
    - Clothing deformation and motion blur/resolution drop.
FR-06: The system shall maintain a consistent Global ID for a person across the entire camera network, updating the ID if a new match is confirmed.

3.3 Data Output and Integration
FR-07: The system shall generate detection metadata containing frame information, bounding box coordinates, local track ID, and the assigned Global ID.
FR-08: The system shall generate cropped images of the detected person, suitable for input into the Cloth Detection module.

4. NON-FUNCTIONAL REQUIREMENTS

4.1 Performance and Latency
NFR-01: Latency Budget Requirements (Measured from input frame to required output/interface update):
    - Detector Module: <= 30 ms per frame.
    - Tracker Module: <= 5 ms per frame.
    - Re-ID Module: <= 8 ms per track feature processing.
NFR-02: Target Throughput: The system should aim to achieve 10 Frames Per Second (FPS) processing capability across the 4 initial camera sources combined on the specified deployment hardware (Intel i5-10500H, RTX 3060).

4.2 Accuracy and Metrics
NFR-03: Re-ID Accuracy: The Multi-Camera Re-ID must meet or exceed the following baseline metrics for reliable deployment:
    - Top-1 Accuracy: >= 85%
    - mAP (Mean Average Precision): >= 80%
NFR-04: General Detection/Tracking Accuracy: Accuracy metrics for detection and single-camera tracking must meet the standards required for practical deployment success (no specific quantitative KPI defined, based on visual inspection and downstream module performance).

4.3 System Architecture and Flexibility
NFR-05: Model Flexibility: The architecture must be designed such that the underlying models for Detection, Tracking, and Re-ID are modular and swappable via defined interfaces without requiring extensive code changes to the core system logic.

4.4 Deployment Environment Constraints
NFR-06: The system must run effectively on the reference hardware: MSI Laptop with Intel Core i5-10500H CPU, 16.0 GB RAM, and NVIDIA GeForce RTX 3060 GPU.

4.5 Data Storage and Retention
NFR-07: The system is only required to store detection metadata related to the tracking process (e.g., timestamps, Global IDs, spatial locations). Long-term retention policies for this data are out of scope for this module, but intermediate storage for Re-ID feature sets must be managed efficiently.

5. USER INTERFACE AND VISUALIZATION

5.1 Frontend Interface
NFR-08: The system frontend shall be developed using Streamlit for operator interaction and monitoring.

5.2 Visualization Requirements
NFR-09: The Streamlit interface must visualize the live or processed video streams, displaying bounding boxes around detected persons.
NFR-10: The bounding boxes must clearly indicate the assigned Local Track ID and the consistent Global ID (if established by Re-ID).

6. INTEGRATION AND INTERFACES

6.1 External Integrations
Integration Requirements: No external system integrations are required for the initial deployment phase beyond receiving the raw CCTV video streams and passing processed data (crops/metadata) to the downstream Cloth Detection module.

6.2 Internal Module Interfaces (As defined by Model Flexibility Requirement)

| Module | Input Interface | Output Interface | Latency Budget (Max) | Swappable? |
| :--- | :--- | :--- | :--- | :--- |
| Detector | Raw Video Frame (Batch) | Standard Detections (BB, Confidence, Class) | <= 30 ms | Yes |
| Tracker | Frame + Detections | Frame + Tracks (Local ID assignment) | <= 5 ms | Yes |
| Re-ID | Track Features | Global ID Assignment | <= 8 ms | Yes |

## Technology Stack
TECH STACK DOCUMENTATION: PERSON DETECTION SYSTEM

1. OVERVIEW

This document outlines the recommended technology stack for the Person Detection System, designed for deployment across multiple CCTV sources within a school campus block. The architecture emphasizes modularity, performance (targeting $\approx 10$ FPS across 4 sources given hardware constraints), and the flexibility to easily swap core computer vision models (Detection, Tracking, and Re-ID).

2. CORE COMPUTING AND LANGUAGE

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| Primary Language | Python 3.10+ | Industry standard for ML/CV development, vast ecosystem support (PyTorch, OpenCV). |
| Deep Learning Framework | PyTorch | Highly flexible, preferred for state-of-the-art research models, and offers good GPU utilization required for low-latency inference. |
| Inference Optimization | ONNX Runtime or TorchScript (via Torch.jit) | Essential for achieving low inference latency ($\leq 30$ ms for Detection) on the constrained hardware (i5-10500H + RTX 3060). |

3. COMPUTER VISION MODULES (Detection, Tracking, Re-ID)

The stack is designed around swappable modules interfacing via standardized data structures (Standard Detections, Tracks).

| Module | Recommended Framework/Model | Justification/Interface | Latency Target |
| :--- | :--- | :--- | :--- |
| **Detection** | YOLOv8 (or custom equivalent) | High performance, speed/accuracy trade-off suitable for edge deployment. Must output standard bounding box format. | $\leq 30$ ms |
| **Tracking (Single Camera)** | DeepSORT or StrongSORT | Established trackers that manage ID persistence within a single camera view. Input: Frame + Detections. Output: Tracks with temporary IDs. | $\leq 5$ ms |
| **Re-Identification (Re-ID)** | OSNet or modified StrongSORT Re-ID backbone (e.g., using a feature extractor like ResNet/EfficientNet trained on Re-ID datasets) | Must provide robust feature embeddings capable of handling viewpoint/pose changes between cameras (Target Top-1 $\geq 85\%$, mAP $\geq 80\%$). Input: Track features. Output: Global Persistent ID. | $\leq 8$ ms |

4. MULTI-CAMERA MANAGEMENT AND DATA FLOW

Since the system handles inputs from 4 sources concurrently and needs to correlate detections across them for Re-ID, robust stream handling is critical.

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| Video Handling | OpenCV (cv2) or specialized stream handlers (e.g., GStreamer bindings if high-throughput IPC is needed) | Standard library for frame grabbing and basic manipulation. Managing 4 sources concurrently requires careful threading/multiprocessing. |
| Concurrency Management | Python `concurrent.futures` (ThreadPoolExecutor/ProcessPoolExecutor) | To manage the I/O bound task (reading streams) and CPU/GPU bound tasks (Inference) across the 4 cameras simultaneously, aiming for the 10 FPS goal. |

5. USER INTERFACE AND MONITORING

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| Frontend/Admin Interface | Streamlit | Directly specified requirement. Provides rapid development for operational control and real-time monitoring (bounding box visualization). |
| Visualization | OpenCV drawing functions integrated into Streamlit updates. | Used for rendering bounding boxes, tracking IDs, and system status overlays during monitoring. |

6. DATA STORAGE AND METADATA

The primary data requirement is storing metadata related to successful detections and ID assignments for auditing and potential downstream integration with the cloth detection module.

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| Metadata Storage | SQLite (for initial deployment) | Lightweight, file-based database suitable for development/testing on a single machine (MSI Laptop). Sufficient for storing detection metadata (Timestamp, Camera ID, Global ID, Bounding Box coordinates). |
| Scalability Note | If the data volume grows significantly, migration to PostgreSQL or a NoSQL solution (like MongoDB) might be considered, but SQLite is adequate for the initial scope (4 sources). |

7. HARDWARE CONSIDERATIONS & OPTIMIZATION

The stack must be tailored to the development hardware: Intel i5-10500H, 16GB RAM, RTX 3060.

*   **GPU Acceleration:** All deep learning models (Detector, Re-ID feature extraction) *must* be deployed using CUDA acceleration on the RTX 3060.
*   **Batching:** Inference on the Detection model should leverage batching where possible (e.g., processing frames from the 4 cameras in a small batch if latency budgets allow) to maximize GPU utilization, while ensuring the end-to-end latency target ($\leq 30$ ms for detection) is met.
*   **Tracking/Re-ID Buffer:** The Tracker and Re-ID modules must be extremely fast ($\leq 5$ ms and $\leq 8$ ms respectively) to avoid bottlenecking the overall pipeline, as they operate on every detected tracklet across all frames.

## Project Structure
PROJECTSTRUCTURE: Person Detection System (School Campus Surveillance)

1. ROOT DIRECTORY: /person_detection_system

2. CORE APPLICATION & CONFIGURATION: /person_detection_system

    2.1. /src: Contains all primary source code modules.

        2.1.1. /modules: Contains modular, swappable implementation logic for detection, tracking, and Re-ID.

            2.1.1.1. /detector: Contains implementations for the object detection model. (Swappable)
                /detector/yolov8_impl.py: Current implementation using YOLOv8 architecture.
                /detector/detector_interface.py: Abstract base class defining the required detection interface (Batch -> Standard Detections).

            2.1.1.2. /tracker: Contains implementations for single-camera tracking. (Swappable)
                /tracker/deep_sort_impl.py: Current implementation (e.g., DeepSORT).
                /tracker/tracker_interface.py: Abstract base class defining the required tracking interface (Frame + Detections -> Tracks).

            2.1.1.3. /reid: Contains implementations for Multi-Camera Re-Identification. (Swappable)
                /reid/osnet_reid_impl.py: Current Re-ID model implementation (e.g., OSNet based).
                /reid/reid_interface.py: Abstract base class defining the required Re-ID interface (Track -> GlobalID).

        2.1.2. /camera_manager: Handles stream ingestion and calibration data management for multiple cameras (Target: 4 sources).
            /camera_manager/stream_handler.py: Logic to ingest RTSP/local streams concurrently.
            /camera_manager/camera_config.json: Stores metadata per camera (e.g., Camera ID, RTSP URL, intrinsic/extrinsic parameters if needed for advanced homography/perspective, although primarily used for simple mapping here).

        2.1.3. /pipeline: Orchestrates the flow between detection, tracking, and Re-ID modules.
            /pipeline/system_orchestrator.py: Main processing loop managing data flow across modules, ensuring latency targets are met.

        2.1.4. /utils: Helper functions.
            /utils/data_structures.py: Defines standard data formats (e.g., DetectionFormat, TrackFormat, GlobalTrackFormat).
            /utils/metrics_logger.py: Tools for logging performance metrics (latency per stage).

    2.2. /frontend: Contains code for the Streamlit user interface.
        /frontend/app.py: Main Streamlit application entry point for monitoring and operation.
        /frontend/visualization_tools.py: Functions dedicated to drawing bounding boxes and assigned Global IDs onto the video streams for visualization.

    2.3. /models: Stores pre-trained weights and necessary auxiliary files.
        /models/detector_weights.pt: Trained/pretrained weights for the active detection model.
        /models/reid_weights.pth: Trained/pretrained weights for the active Re-ID model.
        /models/model_metadata.yaml: Configuration file specifying which implementation (e.g., yolov8_impl) to load at startup.

    2.4. /data: Stores input and output data artifacts during testing/development.

        2.4.1. /input_streams: Directory for static video files used for initial testing/benchmarking if live streams are unavailable.

        2.4.2. /metadata_output: Storage for generated detection metadata (as per data storage requirements).
            /metadata_output/YYYYMMDD_HHMMSS_metadata.csv: Stores records including Timestamp, CameraID, LocalTrackID, GlobalID, BoundingBox coordinates.

    2.5. /configs: System-wide configuration files.
        /configs/system_settings.yaml: Global settings (e.g., GlobalID persistence timeout, Re-ID confidence thresholds for deployment).
        /configs/latency_budgets.yaml: Defines latency targets for each module (Detector ≤30ms, Tracker ≤5ms, Re-ID ≤8ms).

    2.6. ENVIRONMENT & DOCUMENTATION
        requirements.txt: Python dependencies list (e.g., PyTorch, OpenCV, Streamlit, specific model libraries).
        README.md: High-level project overview, setup instructions, and deployment notes specific to the MSI hardware constraints (aiming for 10 FPS @ 4 sources).
        LICENSE: Licensing information.

## Database Schema Design
SCHEMADESIGN: Person Detection System (School Campus)

1. Overview and Goals

This database schema is designed to store the necessary metadata generated by the Person Detection, Tracking, and Multi-Camera Re-Identification (Re-ID) pipeline. The primary goal is to maintain accurate, temporally linked records of detected individuals across multiple CCTV feeds for subsequent processing (e.g., cropping for the cloth detection module) and administrative monitoring.

2. Entity-Relationship Diagram (Conceptual Mapping)

The core entities are CAMERA, DETECTION, TRACK, and PERSON (Global ID).

3. Detailed Schema Design

The system will utilize four primary tables: Camera, Detections, Tracks, and GlobalIdentities.

3.1. Table: Camera

Stores static metadata about each CCTV source.

| Field Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| camera\_id | INT | PK, NOT NULL | Unique identifier for the camera (e.g., 1, 2, 3, 4). |
| camera\_name | VARCHAR(100) | UNIQUE, NOT NULL | Human-readable name (e.g., \"BlockA\_Entrance\_1\"). |
| location\_description | TEXT | NULLABLE | Detailed physical location within the campus. |
| resolution\_x | INT | NOT NULL | Horizontal resolution captured. |
| resolution\_y | INT | NOT NULL | Vertical resolution captured. |
| ip\_address | VARCHAR(50) | NULLABLE | Network address for reference. |
| is\_active | BOOLEAN | NOT NULL | Status of the camera feed. |

3.2. Table: Detections

Stores the raw output from the detection model for every frame processed. This links directly to the frame processing results.

| Field Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| detection\_id | BIGINT | PK, NOT NULL | Unique ID for this specific detection instance. |
| frame\_timestamp | DATETIME | NOT NULL | Exact time the frame was processed. |
| camera\_id | INT | FK (Camera) | Which camera generated this detection. |
| frame\_sequence\_num | INT | NOT NULL | Sequence number within the video stream for this camera. |
| object\_class | VARCHAR(50) | NOT NULL | Detected class (Expected: 'person'). |
| confidence\_score | FLOAT | NOT NULL | Detection model confidence (0.0 to 1.0). |
| bbox\_x1 | INT | NOT NULL | Bounding box top-left X coordinate. |
| bbox\_y1 | INT | NOT NULL | Bounding box top-left Y coordinate. |
| bbox\_x2 | INT | NOT NULL | Bounding box bottom-right X coordinate. |
| bbox\_y2 | INT | NOT NULL | Bounding box bottom-right Y coordinate. |

3.3. Table: Tracks

Stores the output from the single-camera tracking model, linking detections over short temporal sequences within one camera feed.

| Field Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| track\_id | BIGINT | PK, NOT NULL | Unique ID assigned by the single-camera tracker. |
| detection\_id | BIGINT | FK (Detections), UNIQUE | Links to the specific detection instance. |
| frame\_timestamp | DATETIME | NOT NULL | Timestamp (redundant, but useful for indexing). |
| camera\_id | INT | FK (Camera) | Camera source. |
| track\_duration\_frames | INT | NOT NULL | Length of the track segment associated with this detection. |
| reid\_feature\_vector | BLOB/ARRAY | NULLABLE | The feature embedding generated by the Re-ID model for this specific track segment (critical for Re-ID matching). Stored as BLOB or specialized array type based on DB implementation. |
| is\_new\_track | BOOLEAN | NOT NULL | Flag indicating if this is the start of a new track segment. |

3.4. Table: GlobalIdentities (Re-ID Output)

Stores the persistent, system-wide identity assigned by the Multi-Camera Re-ID module, linking multiple track segments across different cameras.

| Field Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| global\_person\_id | UUID/BIGINT | PK, NOT NULL | The unique, persistent ID for a specific person across the entire system. |
| track\_id | BIGINT | FK (Tracks), UNIQUE, NOT NULL | The single-camera track segment this identity was matched against. |
| camera\_id | INT | FK (Camera) | The camera where this track segment occurred. |
| assignment\_timestamp | DATETIME | NOT NULL | When this global ID was assigned/confirmed for this track. |
| reid\_confidence\_score | FLOAT | NOT NULL | Confidence score of the Re-ID matching process (Top-1/mAP related metric). |
| is\_confirmed | BOOLEAN | NOT NULL | True if the ID assignment has passed the configured Re-ID threshold. |
| next\_module\_status | VARCHAR(50) | NOT NULL | Status for the next module (e.g., 'READY\_FOR\_CROP', 'PENDING\_REVIEW'). |
| crop\_metadata\_ref | VARCHAR(255) | NULLABLE | Reference pointer to where the cropped image data is stored externally (if applicable). |

4. Relationships Summary

*   **Camera (1) to Detections (Many):** One camera generates many detection events.
*   **Detections (1) to Tracks (1):** Each detection instance contributes to exactly one single-camera track segment.
*   **Tracks (1) to GlobalIdentities (1):** Each track segment is mapped to one (or potentially zero, if unmatched) global identity. (Note: A single Global ID will map to many Track IDs across time/location).

5. Model Flexibility and Interface Considerations

*   **Model Swapping:** The schema design isolates the feature embedding generation (Re-ID) into the `reid_feature_vector` field within the `Tracks` table. As long as the new Re-ID module adheres to the storage format (BLOB/Array size) defined here, the interface remains stable.
*   **Latency Isolation:** The storage of bounding boxes (`Detections`) and track assignments (`GlobalIdentities`) is decoupled from the latency-critical processing steps (Detector $\le 30$ms, Tracker $\le 5$ms, Re-ID $\le 8$ms). Database insertion happens *after* these processes generate their metadata outputs.
*   **Streamlit Interface:** The data presented to the Streamlit application for monitoring will primarily join `GlobalIdentities`, `Tracks`, and `Camera` tables to visualize the history (`global\_person\_id`) across spatial locations (`camera\_id`) with associated bounding box details from the `Detections` table upon request.

6. Data Retention Policy

Only detection metadata is stored. Raw video streams are assumed to be handled by external VMS or not retained. The retention policy for this metadata must be defined by administrators, though the structure supports long-term archival given the modest storage footprint per event (primarily coordinates, timestamps, and feature vectors).

## User Flow
USERFLOW DOCUMENT: PERSON DETECTION SYSTEM

1. INTRODUCTION AND SCOPE

1.1 Document Purpose
This document details the user flows, interaction patterns, and functional journeys for the Person Detection system, which serves as a prerequisite module feeding into the downstream Cloth Detection module. The primary interface for administrators will be a Streamlit application.

1.2 Target Users
Administrators: Responsible for system operation, configuration, monitoring, and ensuring data integrity.

1.3 System Context
This module handles multi-camera person detection, tracking within a single camera view, and critical Multi-Camera Re-Identification (Re-ID) to assign consistent Global IDs across different camera feeds.

2. CORE USER FLOWS (ADMINISTRATOR INTERACTION)

2.1 Flow 1: System Initialization and Monitoring

Actors: Administrator
Goal: Verify system status and observe real-time detection feeds.

| Step | User Action (Streamlit Interface) | System Response/Interaction Pattern | Notes/Wireframe Description |
| :--- | :--- | :--- | :--- |
| 1.1 | Access Streamlit Application URL (Login/Access Control assumed external or minimal) | Loads Dashboard View. Displays system health indicators. | Dashboard Tab: System Status (Operational/Offline), Latency/FPS summary per camera source. |
| 1.2 | Navigate to "Camera Feeds" tab. | System initiates connection to defined RTSP/Video streams (4 sources initially). | Tab View: Grid layout (2x2) showing live feeds. |
| 1.3 | Monitor live feeds. | Real-time bounding box overlay visualization appears on detected persons. Global IDs are displayed adjacent to bounding boxes if Re-ID has assigned one. | Visualization Detail: Bounding boxes must respect the specified visualization requirements (color-coded or labeled). |
| 1.4 | Observe Re-ID behavior. | System displays a small notification/log entry when a track transitions between cameras (i.e., new Global ID assigned or existing Global ID confirmed). | Backend Logging: Log tracks assigned/re-identified Global IDs. |
| 1.5 | Check performance metrics (optional view). | Displays rolling average FPS (target 10 FPS) and latest detection/Re-ID metric scores (for diagnostics). | Configuration Tab (Read-Only): Performance Metrics display. |

2.2 Flow 2: Model Configuration and Swapping (Flexibility Requirement)

Actors: Administrator
Goal: Update or swap the underlying AI models (Detector, Tracker, Re-ID).

| Step | User Action (Streamlit Interface) | System Response/Interaction Pattern | Notes/Wireframe Description |
| :--- | :--- | :--- | :--- |
| 2.1 | Navigate to "Model Configuration" tab. | Displays current active model names/versions for Detector, Tracker, and Re-ID modules. | Configuration Tab: Section for each module with a dropdown or file upload selector. |
| 2.2 | Select a new Detector Model (e.g., swap from YoloV5 to YoloV8 structure). | User selects the new model artifact (file upload or pre-registered path). | Interface must clearly indicate the required input/output format compatibility (Standard Detections format required). |
| 2.3 | Click "Apply Changes & Test Latency". | System loads the new model into memory for testing. Runs a short inference batch against a test stream. | System checks latency (<30ms for Detector). If failed, displays an error message and reverts load. |
| 2.4 | Confirm deployment of new model. | Upon successful test, the user confirms. The system replaces the active module instance. | A confirmation dialog is essential: "Deploy new [Module Name]? This may cause brief service interruption." |
| 2.5 | Repeat steps for Tracker (latency check <5ms) and Re-ID (latency check <8ms). | Successful loading and validation of new models for tracking and Re-ID logic. | Re-ID confidence threshold tuning option should be available here (e.g., slider input for Re-ID threshold). |

2.3 Flow 3: Data Monitoring and Output Verification (Filtering for Next Module)

Actors: Administrator
Goal: Verify that the output stream (metadata/cropped images) sent to the Cloth Detection module is correct.

| Step | User Action (Streamlit Interface) | System Response/Interaction Pattern | Notes/Wireframe Description |
| :--- | :--- | :--- | :--- |
| 3.1 | Navigate to "Output Verification" tab. | Displays a queue or log of recent successful track completions that met criteria for cropping/forwarding. | Verification Tab: A searchable log interface. |
| 3.2 | Filter view by Camera ID or Global ID. | The system queries the stored detection metadata logs based on the filter. | Search Bar/Dropdowns are necessary due to expected high volume of metadata. |
| 3.3 | Click on a specific metadata entry (which implies a person segment). | System retrieves the associated cropped image data (if stored briefly) and the final assigned Global ID. | Detail View: Shows bounding box coordinates, timestamp, and the assigned Global ID used for the crop. |
| 3.4 | Verify Multi-Camera Consistency (Mental Check). | Administrator checks logs for a known person moving across Camera A and Camera B, ensuring they share the same Global ID, despite changes in pose/lighting (validating Re-ID robustness). | Focus on entries where Re-ID confidence scores are high/low to test thresholding effect. |

3. INTERACTION PATTERNS AND DESIGN CONSIDERATIONS

3.1 Visualization Pattern (Bounding Boxes and IDs)
The primary interaction visualization is the overlay on the CCTV feed.
*   **Color Coding:** Bounding boxes should ideally be color-coded uniquely per active Global ID to aid visual tracking across the grid view, although this may impact performance slightly. If color stability is an issue, simple numbered labels are required.
*   **ID Placement:** Global IDs must be clearly legible outside or immediately adjacent to the bounding box.
*   **Occlusion Handling:** When a person is heavily occluded, the bounding box should remain visible but perhaps flash or change opacity to indicate low confidence detection status.

3.2 Multi-Camera Re-ID Interaction Fidelity
Since Re-ID is crucial, the interface must facilitate testing its robustness without requiring deep ML knowledge:
*   **Confidence Display:** For any track confirmed via Re-ID transfer (A -> B), the system must visually indicate the Re-ID confidence score used for the match (e.g., "ID: G102 [Conf: 91%]")—especially important during tuning.
*   **ID Conflict Log:** A specific alert log should capture instances where the system *failed* to match tracks that should logically be the same person (False Negative Re-ID) or incorrectly matched different people (False Positive Re-ID).

3.3 Latency Feedback Loop
Due to strict latency budgets (Detector ≤30ms, Tracker ≤5ms, Re-ID ≤8ms), the Streamlit interface must provide immediate, transparent feedback on processing delays:
*   **Real-Time Latency Tickers:** Small indicators on the dashboard showing the average time taken for the most recent inference cycle for each core module.
*   **Hardware Monitoring:** Simple visualization of CPU/GPU utilization on the host machine (MSI i5-10500H/RTX 3060) to help administrators correlate high latency with hardware bottlenecks.

4. DATA FLOW AND METADATA STRUCTURE (Implicit Interaction)

The administrator interacts primarily through observing the results of this automated flow:

Camera Sources (4 streams) -> Detector Module (Batch Processing, Output: Detections) -> Tracker Module (Frame + Detections, Output: Tracks) -> Re-ID Module (Track Embeddings -> Global ID Assignment) -> Data Store (Detection Metadata Storage) -> Cloth Detection Module (Output: Cropped Data + Global ID).

The administrator's operational view is monitoring the integrity between the Tracker output and the Re-ID assignment into the Data Store.

## Styling Guidelines
STYLING GUIDELINES DOCUMENT: PERSON DETECTION MODULE

1. INTRODUCTION AND PURPOSE

This document outlines the styling, visual design, and user interface (UI/UX) principles for the Person Detection module, intended for use within a Streamlit-based administrative dashboard for campus security monitoring. The primary goal is to ensure a professional, high-contrast, and information-dense interface suitable for continuous monitoring and system operation by administrators.

2. TARGET AUDIENCE AND CONTEXT

Target Users: System Administrators.
Context: Monitoring real-time processing status, configuration, and reviewing detection metadata. The interface must prioritize clarity and performance visualization over aesthetic flair. Given the technical nature and latency requirements, the design should be clean, minimizing visual clutter.

3. COLOR PALETTE

The color palette is selected for high contrast, readability under prolonged viewing, and clear indication of system status, optimized for monitoring environments.

Primary Color (Background/Base): Dark Theme Recommended
Rationale: Reduces eye strain during long monitoring sessions.

| Color Name | Hex Code | Usage |
|---|---|---|
| Primary Background | #1A1A1A (Deep Charcoal) | Main dashboard canvas, panel backgrounds. |
| Secondary Background | #2C2C2C (Dark Gray) | Card backgrounds, segmented controls. |
| Primary Text/Foreground | #E0E0E0 (Off-White) | Body text, labels, default status indicators. |
| Accent/Primary Action | #007ACC (Vibrant Blue) | Buttons, active selections, system labels. |
| Success/Clear Status | #38A169 (System Green) | Confirmation messages, models running normally. |
| Warning/Caution | #F6AD55 (Amber/Orange) | Non-critical warnings (e.g., temporary latency spikes). |
| Alert/Error | #E53E3E (System Red) | Critical failures, connection loss, high error rates. |

4. TYPOGRAPHY

The typography prioritizes legibility and consistent hierarchy across monitoring metrics and configuration panels. Streamlit's default font family (typically sans-serif like Roboto or similar system fonts) is acceptable, but size and weight must be managed strictly.

| Element | Font Family | Weight | Size (Relative) | Usage Context |
|---|---|---|---|---|
| Headings (H1, Module Title) | Sans-serif | Bold (700) | Large (1.75em) | Main dashboard title, major section dividers. |
| Subheadings (H2, Panel Titles) | Sans-serif | Semi-Bold (600) | Medium (1.3em) | Widget titles (e.g., "Camera Feed Status"). |
| Body Text/Labels | Sans-serif | Regular (400) | Standard (1.0em) | Descriptions, metadata labels. |
| Metric Values/Data | Sans-serif | Medium (500) | Slightly Larger (1.1em) | KPI readouts, latency figures, ID counts. |
| Code/Technical Data | Monospace (e.g., Courier New) | Regular (400) | Small (0.9em) | Model versions, configuration paths. |

5. UI/UX PRINCIPLES

5.1. Information Density and Hierarchy
The primary goal is monitoring 4 camera sources and the overall system health. The layout must be information-dense without feeling cluttered.
*   **Layering:** Critical status (e.g., Alert/Error) must always override less critical information (e.g., background color).
*   **Grouping:** Related metrics (e.g., Detector latency, Tracker latency, Re-ID latency) must be visually grouped together using secondary background cards.

5.2. Visualization of Detection and Tracking
Visual feedback is crucial for administrators to trust the system.

*   **Bounding Box Visualization:** Bounding boxes overlaid on live (or recent snapshot) CCTV feeds must adhere to the following:
    *   **Default Detection:** Thin, solid border using Accent Blue (#007ACC).
    *   **Tracked Object (Stable ID):** Thicker border using Success Green (#38A169). The ID number should be displayed directly adjacent to the box in an Off-White color with a small, dark background box for maximum readability.
    *   **Re-ID Confirmation (High Confidence):** Temporary visual flicker or color shift (e.g., to Cyan) when a track successfully crosses cameras, reinforcing the Re-ID success.
    *   **Occlusion/Uncertainty:** If tracking confidence drops significantly, the bounding box border should switch to Warning Yellow (#F6AD55).

5.3. System Status Indicators
System status must be immediately identifiable, especially concerning the low latency budgets required (Detector $\le 30ms$, Tracker $\le 5ms$, Re-ID $\le 8ms$).

*   **Latency Monitoring:** Latency gauges or numerical readouts must visually compare current performance against the defined budget.
    *   If latency is within 90% of the budget: Displayed in standard text color.
    *   If latency exceeds 90% of the budget: Text turns Warning Yellow.
    *   If latency exceeds 100% of the budget: Text turns Alert Red, potentially accompanied by a status flag.

5.4. Model Flexibility and Configuration Interface
Since models are swappable, configuration panels must be unambiguous.

*   **Selection Controls:** Dropdowns or radio buttons for selecting Detector, Tracker, or Re-ID models should use the Accent Blue for the currently active selection.
*   **Metadata Display:** The currently loaded model name, version (if applicable), and associated performance metrics (Top-1/mAP for Re-ID) must be prominently displayed near the configuration area.

5.5. Inter-Camera Consistency (Re-ID Feedback)
The visualization must reflect the success of Multi-Camera Re-ID.

*   **Global ID Assignment:** When a person is consistently tracked across multiple cameras (retaining a Global ID), this ID should persist visibly in the monitoring panel associated with that track, perhaps with a small icon indicating "Multi-Camera Verified."
*   **ID Stability Visualization:** A metric showing the percentage of time active IDs have remained stable over the last 5 minutes should be displayed clearly, ideally using a gauge or progress bar, leveraging the Success Green for stability.

6. LAYOUT GUIDELINES (Streamlit Implementation Focus)

The interface should follow a structure that allows for quick scanning:

1.  **Top Bar (System Health):** Global status indicators (System OK/Fail), overall FPS, and global latency averages. Use minimal text and large, color-coded icons.
2.  **Left Sidebar (Configuration/Controls):** Model selection, input stream management (Camera selection), and metadata export controls. Keep this section clean and focused on inputs.
3.  **Main Area (Monitoring Views):**
    *   **Primary View:** Grid layout for the 4 camera feeds, with bounding boxes and track IDs overlayed according to Section 5.2.
    *   **Secondary View (Metrics Panel):** Dedicated area for real-time latency graphs, Re-ID success rates, and detection metadata logs. This panel should be scrollable if detailed metadata exceeds screen height.

7. ACCESSIBILITY CONSIDERATIONS

All text must meet a minimum contrast ratio of 4.5:1 against its background color, particularly when using the dark theme and vibrant accent colors. Color alone should not be the sole indicator of status; accompanying text labels (e.g., "Alert," "OK") or distinct iconography must be used alongside color changes.
