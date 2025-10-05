# Project README

## Team Information
**Team Name: Helloworld**  
**Team ID: T132**  

### Team Members & Roles
| Member Name   | Role |
|---------------|------|
| Ishan Kashyap | Leader |
| Varun Kumar   | Member |
| Nishant       | Member |
| Sneha Gupta   | Member |
| Uday          | Member |

## Problem Statement
Currently, Situational Reports (SITREPs) are created manually, either written or dictated by soldiers under high-pressure conditions. Continuous monitoring of surveillance feeds leads to fatigue and increases the likelihood of missing critical details. Accurately identifying emotions, behaviors, or potential hostile actions without technological assistance is challenging. The core problem is the lack of automated, real-time tools that can generate SITREPs, analyze threats, and predict suspicious activities, which limits the ability of soldiers to make faster, safer, and more reliable operational decisions.

## Tech Stack
- **APP Frontend: Flutter** 
- **WEB Frontend: HTML, CSS, JavaScript, Flask** 
- **Backend: python**   
- **Database: firebase**
- **detection: mediapipe, OpenCV, Ultralytics**
- **Other Tools/Technologies: Ollama, Gemma, Json** 

## How to Run the Project
### Prerequisites (Web)
- python should be installed
- python libraries OpenCV-python, mediapipe, Ultralytics, Numpy,Torch, Flask, Streamlit, os, json, cv2, pathlib, threading, queue, torchvision 

### Prerequisites (App)
- flutter sdk
- Visual Studio Code

### quick checklist agr phone mein clana hai
- Edit backend.py → app.run(host='0.0.0.0', port=5000).
- Start python backend.py.
- On computer run ipconfig/ifconfig → copy LAN IP (say 192.168.1.42).
- In homepage.dart set _baseUrl = 'http://192.168.1.42:5000'.
- Ensure firewall allows port 5000.
- Put phone on same Wi-Fi and open http://192.168.1.42:5000/get_sitrep in mobile browser to test.
- Run the Flutter app on the phone — try upload a small video. (phone mein vidio ka size zyada hota hai toh km he bnana)

### Setup Instructions
1. Clone the repository:  
   ```bash
   git clone https://github.com/Nishant-dev-byte/HELLOWORLD.git
   cd HELLOWORLD
   install all required python libraries

   python backend.py in terminal

   install live-server extension in vs code
   use go live in loginpage.html 

   enter id = abc@gdg.com
   enter password =111222

   now you can run the web app
