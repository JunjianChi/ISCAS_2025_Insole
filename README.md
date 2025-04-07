# ISCAS_2025_Insole

This project focuses on capturing high-resolution plantar pressure data from a custom insole equipped with **255 pressure sensors**. The system is designed for gait analysis and motion recognition using embedded hardware and machine learning pipelines.

<br/>

<img src="./mdfile/Insole_movie.gif" width="600" height="350"/>

---

## 🧠 Project Overview

- **Hardware**: The insole includes two 32×32 analog multiplexer arrays to capture foot pressure data across 255 points.
- **Data Acquisition**: Data is read using an ESP32E microcontroller and mapped into a 2D grid format.
- **ML Integration**: The captured pressure maps are synchronized with depth camera data for machine learning-based classification.
- **Application**: This system is ideal for gait phase detection, posture evaluation, or medical diagnostics.

---

## 📁 Repository Structure

| Folder / File              | Description                                  |
|----------------------------|----------------------------------------------|
| `3d_model/`                | Enclosure CAD files for the insole           |
| `Experiment_Dataset/`      | Raw and processed sensor datasets            |
| `PCB/`                     | Circuit board design files                   |
| `Software_Code/`           | Host PC-side software for data processing    |
| `esp32E_insole_code/`      | ESP32 firmware for sensor scanning           |
| `mdfile/`                  | Media files (e.g. GIF, figures)              |
| `ISCAS_Insole_Paper.pdf`   | Final version of the ISCAS 2025 paper        |
| `Live_demo_Insole_final.pdf`| Demo or poster for conference presentation  |
| `README.md`                | This documentation file                      |

---

## 🛠️ Development Environment

- ESP-IDF v5.3.0 (for ESP32 development)
- Python (for data post-processing & ML)
- KiCad / Altium Designer (for PCB)
- Fusion360 / SolidWorks (for 3D model design)

---

## 🔒 License

MIT License 

---


