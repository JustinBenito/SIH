# import cv2
# import mediapipe as mp
# import time
# import numpy as np
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi import FastAPI, WebSocket
# import asyncio 

# app = FastAPI()

# origins = ["http://localhost:3000"]  # Add the frontend URL here
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize Mediapipe components
# # mp_drawing = mp.solutions.drawing_utils
# # drawing_spec=mp_drawing.DrawingSpec(thickness=1,circle_radius=1)
# mp_pose = mp.solutions.pose
# mp_face_mesh = mp.solutions.face_mesh

# # disturbed=False

# # Initialize the video capture
# # cap = cv2.VideoCapture(1)


# def process_frame_with_model(frame):
#     disturbed = False
# # Initialize the Face Mesh and Pose models
#     with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
#         mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

#         closed_eye_counter = 0
#         head_orientation_timer = 0
#         head_orientation_direction = None
#         start_time = None
#         nose_3d = np.array([0, 0, 0], dtype=np.float64) 
#         nose_2d=np.array([0,0])

#         while frame.isOpened():
#             ret, frame = frame.read()
#             if not ret:
#                 break

#             # Convert the frame to RGB
#             frame_rgb = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB)

#             frame_rgb.flags.writeable=False


#             # Process the frame with Face Mesh
#             face_results = face_mesh.process(frame_rgb)
#             pose_results = pose.process(frame_rgb)
#             frame_rgb.flags.writeable=True

#             image=cv2.cvtColor(frame_rgb,cv2.COLOR_RGB2BGR)
#             img_h,img_w,img_c=image.shape
#             face_3d=[]
#             face_2d=[]

#             if face_results.multi_face_landmarks:
#                 for face_landmarks in face_results.multi_face_landmarks:
#                     for idx,lm in enumerate(face_landmarks.landmark):
#                         if idx==33 or idx==263 or idx==61 or idx==291 or idx==199:
#                             if idx==1:
#                                 nose_2d=(lm.x*img_w,lm.y*img_h)
#                                 nose_3d=(lm.x*img_w,lm.y*img_h,lm.z*3000)
#                             x,y=int(lm.x*img_w),int(lm.y*img_h)

#                             face_2d.append([x,y])

#                             face_3d.append([x,y,lm.z])
                    
#                     face_2d=np.array(face_2d,dtype=np.float64) 
#                     face_3d=np.array(face_3d,dtype=np.float64)

#                     focal_length=1*img_w

#                     cam_matrix=np.array([[focal_length,0,img_h/2],
#                                         [0,focal_length,img_w/2],
#                                         [0,0,1]])

#                     dist_matrix=np.zeros((4,1),dtype=np.float64)

#                     sucess,rot_vec,trans_vec=cv2.solvePnP(face_3d,face_2d,cam_matrix,dist_matrix)
#                     rmat,jac=cv2.Rodrigues(rot_vec)

#                     angles,mtxR,mtxQ,Qx,Qy,Qz=cv2.RQDecomp3x3(rmat)

#                     x=angles[0]*360
#                     y=angles[1]*360
#                     z=angles[2]*360

#                     if y < -10:
#                         text="looking left"
#                         # print("left")
#                         disturbed=True
#                     elif y>10:
#                         text="looking right"
#                         # print("right")
#                         disturbed=True
#                     elif x< -10:
#                         text="looking down"
#                         # print("down")
#                     elif x > 10:
#                         text="Looking up"
#                         # print("up")
#                         disturbed=True
#                     else:
#                         text= "Forward"
#                         disturbed=False
#                         # print("forward")

#                     nose_3d_projection,jacobian=cv2.projectPoints(nose_3d, rot_vec,trans_vec,cam_matrix,dist_matrix)

#                     p1=(int(nose_2d[0]),int(nose_2d[1]))
#                     p2=(int(nose_2d[0]+y*10),int(nose_2d[1]-x*10))

#                     cv2.line(image,p1,p2,(255,0,0),3)

#                     cv2.putText(image,text,(20,50),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(0,255,0),2)
#                     cv2.putText(image,'x: '+str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
#                     cv2.putText(image,'y: '+str(np.round(y,2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
#                     cv2.putText(image,'z: '+str(np.round(z,2)),(500,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))

#                     # mp_drawing.draw_landmarks(image=image,landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=drawing_spec,connection_drawing_spec=drawing_spec)




#             if face_results.multi_face_landmarks:
#                 for landmarks in face_results.multi_face_landmarks:
#                     # Extract eye landmarks
#                     left_eye_landmarks = [landmarks.landmark[i] for i in range(159, 145, -1)]
#                     right_eye_landmarks = [landmarks.landmark[i] for i in range(386, 374, -1)]

#                     # Calculate eye aspect ratio (EAR) for left and right eyes
#                     left_eye_ear = (cv2.norm(left_eye_landmarks[1].x - left_eye_landmarks[5].x,
#                                             left_eye_landmarks[1].y - left_eye_landmarks[5].y) +
#                                     cv2.norm(left_eye_landmarks[2].x - left_eye_landmarks[4].x,
#                                             left_eye_landmarks[2].y - left_eye_landmarks[4].y)) / \
#                                 (2 * cv2.norm(left_eye_landmarks[0].x - left_eye_landmarks[3].x,
#                                                 left_eye_landmarks[0].y - left_eye_landmarks[3].y))

#                     right_eye_ear = (cv2.norm(right_eye_landmarks[1].x - right_eye_landmarks[5].x,
#                                             right_eye_landmarks[1].y - right_eye_landmarks[5].y) +
#                                     cv2.norm(right_eye_landmarks[2].x - right_eye_landmarks[4].x,
#                                             right_eye_landmarks[2].y - right_eye_landmarks[4].y)) / \
#                                     (2 * cv2.norm(right_eye_landmarks[0].x - right_eye_landmarks[3].x,
#                                                 right_eye_landmarks[0].y - right_eye_landmarks[3].y))


#                     if left_eye_ear < 0.2 or right_eye_ear < 0.2:
#                         closed_eye_counter += 1
#                     else:
#                         closed_eye_counter = 0

#                     if closed_eye_counter > 15:  # You can adjust this threshold
#                         # cv2.putText(frame, "Sleeping", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                         disturbed=True

#     return disturbed


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
    
#     while True:
#         try:
#             frame_bytes = await websocket.receive_bytes()
#             frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

#             disturbed_result = process_frame_with_model(frame)

#             await websocket.send_text(str(disturbed_result))
#         except:
#             break

# @app.post("/process_frame")
# async def process_frame(frame_bytes: bytes):
#     print(frame_bytes)
#     frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
#     disturbed_result = process_frame_with_model(frame)
#     return {"disturbed": disturbed_result}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


#         # Display the frame
#     #     cv2.imshow("Frame", frame)
#     #     print(disturbed)

#     #     # Exit the loop if 'q' is pressed
#     #     if cv2.waitKey(1) & 0xFF == ord('q'):
#     #         break

#     # # Release the capture and destroy all OpenCV windows when finished
#     # cap.release()
#     # cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import time
# import numpy as np
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi import FastAPI, WebSocket
# import asyncio 
# from fastapi.responses import HTMLResponse
# from fastapi.websockets import WebSocket
# import uvicorn
# from starlette.websockets import WebSocketDisconnect  

# from fastapi import WebSocket

# app = FastAPI()

# connected_websockets = set()

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     connected_websockets.add(websocket)
#     try:
#         while True:
#             # Send 'disturbed' variable to connected clients
#             await websocket.send_text(str(disturbed))
#             await asyncio.sleep(1)  # Adjust the delay as needed
#     except WebSocketDisconnect:
#         connected_websockets.remove(websocket)

# # Initialize Mediapipe components
# mp_drawing = mp.solutions.drawing_utils
# drawing_spec=mp_drawing.DrawingSpec(thickness=1,circle_radius=1)
# mp_pose = mp.solutions.pose
# mp_face_mesh = mp.solutions.face_mesh

# disturbed=False

# # Initialize the video capture
# cap = cv2.VideoCapture(1)

# # Initialize the Face Mesh and Pose models
# with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
#      mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

#     closed_eye_counter = 0
#     head_orientation_timer = 0
#     head_orientation_direction = None
#     start_time = None
#     nose_3d = np.array([0, 0, 0], dtype=np.float64) 
#     nose_2d=np.array([0,0])

#     async def update_disturbed():
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Convert the frame to RGB
#             frame_rgb = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB)

#             frame_rgb.flags.writeable=False


#             # Process the frame with Face Mesh
#             face_results = face_mesh.process(frame_rgb)
#             pose_results = pose.process(frame_rgb)
#             frame_rgb.flags.writeable=True

#             image=cv2.cvtColor(frame_rgb,cv2.COLOR_RGB2BGR)
#             img_h,img_w,img_c=image.shape
#             face_3d=[]
#             face_2d=[]

#             if face_results.multi_face_landmarks:
#                 for face_landmarks in face_results.multi_face_landmarks:
#                     for idx,lm in enumerate(face_landmarks.landmark):
#                         if idx==33 or idx==263 or idx==61 or idx==291 or idx==199:
#                             if idx==1:
#                                 nose_2d=(lm.x*img_w,lm.y*img_h)
#                                 nose_3d=(lm.x*img_w,lm.y*img_h,lm.z*3000)
#                             x,y=int(lm.x*img_w),int(lm.y*img_h)

#                             face_2d.append([x,y])

#                             face_3d.append([x,y,lm.z])
                    
#                     face_2d=np.array(face_2d,dtype=np.float64) 
#                     face_3d=np.array(face_3d,dtype=np.float64)

#                     focal_length=1*img_w

#                     cam_matrix=np.array([[focal_length,0,img_h/2],
#                                         [0,focal_length,img_w/2],
#                                         [0,0,1]])

#                     dist_matrix=np.zeros((4,1),dtype=np.float64)

#                     sucess,rot_vec,trans_vec=cv2.solvePnP(face_3d,face_2d,cam_matrix,dist_matrix)
#                     rmat,jac=cv2.Rodrigues(rot_vec)

#                     angles,mtxR,mtxQ,Qx,Qy,Qz=cv2.RQDecomp3x3(rmat)

#                     x=angles[0]*360
#                     y=angles[1]*360
#                     z=angles[2]*360

#                     if y < -10:
#                         text="looking left"
#                         # print("left")
#                         disturbed=True
#                     elif y>10:
#                         text="looking right"
#                         # print("right")
#                         disturbed=True
#                     elif x< -10:
#                         text="looking down"
#                         # print("down")
#                     elif x > 10:
#                         text="Looking up"
#                         # print("up")
#                         disturbed=True
#                     else:
#                         text= "Forward"
#                         disturbed=False
#                         # print("forward")

#                     nose_3d_projection,jacobian=cv2.projectPoints(nose_3d, rot_vec,trans_vec,cam_matrix,dist_matrix)

#                     p1=(int(nose_2d[0]),int(nose_2d[1]))
#                     p2=(int(nose_2d[0]+y*10),int(nose_2d[1]-x*10))

#                     cv2.line(image,p1,p2,(255,0,0),3)

#                     cv2.putText(image,text,(20,50),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(0,255,0),2)
#                     cv2.putText(image,'x: '+str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
#                     cv2.putText(image,'y: '+str(np.round(y,2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
#                     cv2.putText(image,'z: '+str(np.round(z,2)),(500,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))

#                     mp_drawing.draw_landmarks(image=image,landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=drawing_spec,connection_drawing_spec=drawing_spec)




#             if face_results.multi_face_landmarks:
#                 for landmarks in face_results.multi_face_landmarks:
#                     # Extract eye landmarks
#                     left_eye_landmarks = [landmarks.landmark[i] for i in range(159, 145, -1)]
#                     right_eye_landmarks = [landmarks.landmark[i] for i in range(386, 374, -1)]

#                     # Calculate eye aspect ratio (EAR) for left and right eyes
#                     left_eye_ear = (cv2.norm(left_eye_landmarks[1].x - left_eye_landmarks[5].x,
#                                             left_eye_landmarks[1].y - left_eye_landmarks[5].y) +
#                                     cv2.norm(left_eye_landmarks[2].x - left_eye_landmarks[4].x,
#                                             left_eye_landmarks[2].y - left_eye_landmarks[4].y)) / \
#                                 (2 * cv2.norm(left_eye_landmarks[0].x - left_eye_landmarks[3].x,
#                                                 left_eye_landmarks[0].y - left_eye_landmarks[3].y))

#                     right_eye_ear = (cv2.norm(right_eye_landmarks[1].x - right_eye_landmarks[5].x,
#                                             right_eye_landmarks[1].y - right_eye_landmarks[5].y) +
#                                     cv2.norm(right_eye_landmarks[2].x - right_eye_landmarks[4].x,
#                                             right_eye_landmarks[2].y - right_eye_landmarks[4].y)) / \
#                                     (2 * cv2.norm(right_eye_landmarks[0].x - right_eye_landmarks[3].x,
#                                                 right_eye_landmarks[0].y - right_eye_landmarks[3].y))


#                     if left_eye_ear < 0.2 or right_eye_ear < 0.2:
#                         closed_eye_counter += 1
#                     else:
#                         closed_eye_counter = 0

#                     if closed_eye_counter > 15:  # You can adjust this threshold
#                         # cv2.putText(frame, "Sleeping", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                         disturbed=True

#             cv2.imshow("Frame", frame)

#             for websocket in connected_websockets:
#                 await websocket.send_text(str(disturbed))

#         # Display the frame
#             # cv2.imshow("Frame", frame)
#             print(disturbed)
#         # @app.post("/cam")
#         # def camera():
#         #     return {"disturbed": disturbed}

#     asyncio.create_task(update_disturbed())

#         # Exit the loop if 'q' is pressed


#     # Release the capture and destroy all OpenCV windows when finished
#     cap.release()
#     cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import numpy as np
# from fastapi import FastAPI

# # Initialize Mediapipe components
# mp_drawing = mp.solutions.drawing_utils
# drawing_spec=mp_drawing.DrawingSpec(thickness=1,circle_radius=1)
# mp_pose = mp.solutions.pose
# mp_face_mesh = mp.solutions.face_mesh

# app = FastAPI()

# active_connections = []

# disturbed=False

# # Initialize the video capture
# cap = cv2.VideoCapture(1)

# # Initialize the Face Mesh and Pose models
# with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
#      mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

#     closed_eye_counter = 0
#     head_orientation_timer = 0
#     head_orientation_direction = None
#     start_time = None
#     nose_3d = np.array([0, 0, 0], dtype=np.float64) 
#     nose_2d=np.array([0,0])

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert the frame to RGB
#         frame_rgb = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB)

#         frame_rgb.flags.writeable=False


#         # Process the frame with Face Mesh
#         face_results = face_mesh.process(frame_rgb)
#         pose_results = pose.process(frame_rgb)
#         frame_rgb.flags.writeable=True

#         image=cv2.cvtColor(frame_rgb,cv2.COLOR_RGB2BGR)
#         img_h,img_w,img_c=image.shape
#         face_3d=[]
#         face_2d=[]

#         if face_results.multi_face_landmarks:
#             for face_landmarks in face_results.multi_face_landmarks:
#                 for idx,lm in enumerate(face_landmarks.landmark):
#                     if idx==33 or idx==263 or idx==61 or idx==291 or idx==199:
#                         if idx==1:
#                             nose_2d=(lm.x*img_w,lm.y*img_h)
#                             nose_3d=(lm.x*img_w,lm.y*img_h,lm.z*3000)
#                         x,y=int(lm.x*img_w),int(lm.y*img_h)

#                         face_2d.append([x,y])

#                         face_3d.append([x,y,lm.z])
                
#                 face_2d=np.array(face_2d,dtype=np.float64) 
#                 face_3d=np.array(face_3d,dtype=np.float64)

#                 focal_length=1*img_w

#                 cam_matrix=np.array([[focal_length,0,img_h/2],
#                                      [0,focal_length,img_w/2],
#                                      [0,0,1]])

#                 dist_matrix=np.zeros((4,1),dtype=np.float64)

#                 sucess,rot_vec,trans_vec=cv2.solvePnP(face_3d,face_2d,cam_matrix,dist_matrix)
#                 rmat,jac=cv2.Rodrigues(rot_vec)

#                 angles,mtxR,mtxQ,Qx,Qy,Qz=cv2.RQDecomp3x3(rmat)

#                 x=angles[0]*360
#                 y=angles[1]*360
#                 z=angles[2]*360

#                 if y < -10:
#                     text="looking left"
#                     # print("left")
#                     disturbed=True
#                 elif y>10:
#                     text="looking right"
#                     # print("right")
#                     disturbed=True
#                 elif x< -10:
#                     text="looking down"
#                     # print("down")
#                 elif x > 10:
#                     text="Looking up"
#                     # print("up")
#                     disturbed=True
#                 else:
#                     text= "Forward"
#                     disturbed=False
#                     # print("forward")

#                 nose_3d_projection,jacobian=cv2.projectPoints(nose_3d, rot_vec,trans_vec,cam_matrix,dist_matrix)

#                 p1=(int(nose_2d[0]),int(nose_2d[1]))
#                 p2=(int(nose_2d[0]+y*10),int(nose_2d[1]-x*10))

#                 cv2.line(image,p1,p2,(255,0,0),3)

#                 cv2.putText(image,text,(20,50),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(0,255,0),2)
#                 cv2.putText(image,'x: '+str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
#                 cv2.putText(image,'y: '+str(np.round(y,2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
#                 cv2.putText(image,'z: '+str(np.round(z,2)),(500,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))

#                 mp_drawing.draw_landmarks(image=image,landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=drawing_spec,connection_drawing_spec=drawing_spec)




#         if face_results.multi_face_landmarks:
#             for landmarks in face_results.multi_face_landmarks:
#                 # Extract eye landmarks
#                 left_eye_landmarks = [landmarks.landmark[i] for i in range(159, 145, -1)]
#                 right_eye_landmarks = [landmarks.landmark[i] for i in range(386, 374, -1)]

#                 # Calculate eye aspect ratio (EAR) for left and right eyes
#                 left_eye_ear = (cv2.norm(left_eye_landmarks[1].x - left_eye_landmarks[5].x,
#                                          left_eye_landmarks[1].y - left_eye_landmarks[5].y) +
#                                 cv2.norm(left_eye_landmarks[2].x - left_eye_landmarks[4].x,
#                                          left_eye_landmarks[2].y - left_eye_landmarks[4].y)) / \
#                                (2 * cv2.norm(left_eye_landmarks[0].x - left_eye_landmarks[3].x,
#                                             left_eye_landmarks[0].y - left_eye_landmarks[3].y))

#                 right_eye_ear = (cv2.norm(right_eye_landmarks[1].x - right_eye_landmarks[5].x,
#                                           right_eye_landmarks[1].y - right_eye_landmarks[5].y) +
#                                  cv2.norm(right_eye_landmarks[2].x - right_eye_landmarks[4].x,
#                                           right_eye_landmarks[2].y - right_eye_landmarks[4].y)) / \
#                                 (2 * cv2.norm(right_eye_landmarks[0].x - right_eye_landmarks[3].x,
#                                              right_eye_landmarks[0].y - right_eye_landmarks[3].y))


#                 if left_eye_ear < 0.2 or right_eye_ear < 0.2:
#                     closed_eye_counter += 1
#                 else:
#                     closed_eye_counter = 0

#                 if closed_eye_counter > 15:  # You can adjust this threshold
#                     # cv2.putText(frame, "Sleeping", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                     disturbed=True

#         # Display the frame
#         cv2.imshow("Frame", frame)
#         print(disturbed)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# @app.post("/camera")
# async def camera():
#     return {"disturbed": disturbed}
#         # Exit the loop if 'q' is pressed


    # Release the capture and destroy all OpenCV windows when finished


from fastapi import FastAPI, WebSocket
from llama_index import ServiceContext, StorageContext
from llama_index.readers.web import BeautifulSoupWebReader

from langchain import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index import (GPTVectorStoreIndex, LLMPredictor,
                         PromptHelper, ServiceContext, download_loader)

from fastapi import FastAPI, WebSocket
from typing import Union
import sys
import os
from llama_index import load_index_from_storage

app = FastAPI()



from llama_index.readers.web import BeautifulSoupWebReader
from langchain import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index import (GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, download_loader)
from pathlib import Path
import sys
import os
import openai
from fastapi.middleware.cors import CORSMiddleware
openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = "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"
from llama_index import ServiceContext, StorageContext
from pathlib import Path
from llama_index import download_loader


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

from llama_index import load_index_from_storage
def ask_ai(question):
    storage_context=StorageContext.from_defaults(persist_dir='index.json')
    index=load_index_from_storage(storage_context)
    query_engine=index.as_query_engine()
    response=query_engine.query(question)
    return response
    

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI
import threading
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from typing import Dict
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

app = FastAPI()

origins = ["http://localhost:3000"]  # Add the frontend URL here
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

disturbed = False

def video_processing_loop():
    global closed_eye_counter
    global disturbed
    cap = cv2.VideoCapture(1)
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

            frame_rgb.flags.writeable=False


            # Process the frame with Face Mesh
            face_results = face_mesh.process(frame_rgb)
            # pose_results = pose.process(frame_rgb)
            frame_rgb.flags.writeable=True

            image=cv2.cvtColor(frame_rgb,cv2.COLOR_RGB2BGR)
            img_h,img_w,img_c=image.shape
            face_3d=[]
            face_2d=[]

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    for idx,lm in enumerate(face_landmarks.landmark):
                        if idx==33 or idx==263 or idx==61 or idx==291 or idx==199:
                            if idx==1:
                                nose_2d=(lm.x*img_w,lm.y*img_h)
                                nose_3d=(lm.x*img_w,lm.y*img_h,lm.z*3000)
                            x,y=int(lm.x*img_w),int(lm.y*img_h)

                            face_2d.append([x,y])

                            face_3d.append([x,y,lm.z])
                    
                    face_2d=np.array(face_2d,dtype=np.float64) 
                    face_3d=np.array(face_3d,dtype=np.float64)

                    focal_length=1*img_w

                    cam_matrix=np.array([[focal_length,0,img_h/2],
                                        [0,focal_length,img_w/2],
                                        [0,0,1]])

                    dist_matrix=np.zeros((4,1),dtype=np.float64)

                    sucess,rot_vec,trans_vec=cv2.solvePnP(face_3d,face_2d,cam_matrix,dist_matrix)
                    rmat,jac=cv2.Rodrigues(rot_vec)

                    angles,mtxR,mtxQ,Qx,Qy,Qz=cv2.RQDecomp3x3(rmat)

                    x=angles[0]*360
                    y=angles[1]*360
                    z=angles[2]*360

                    if y < -10:
                        text="looking left"
                        # print("left")
                        disturbed=True
                    elif y>10:
                        text="looking right"
                        # print("right")
                        disturbed=True
                    elif x< -10:
                        text="looking down"
                        # print("down")
                    elif x > 10:
                        text="Looking up"
                        # print("up")
                        disturbed=True
                    else:
                        text= "Forward"
                        disturbed=False
                        # print("forward")

                    # nose_3d_projection,jacobian=cv2.projectPoints(nose_3d, rot_vec,trans_vec,cam_matrix,dist_matrix)

                    # p1=(int(nose_2d[0]),int(nose_2d[1]))
                    # p2=(int(nose_2d[0]+y*10),int(nose_2d[1]-x*10))

                    # cv2.line(image,p1,p2,(255,0,0),3)

                    # cv2.putText(image,text,(20,50),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(0,255,0),2)
                    # cv2.putText(image,'x: '+str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
                    # cv2.putText(image,'y: '+str(np.round(y,2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
                    # cv2.putText(image,'z: '+str(np.round(z,2)),(500,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))

                    # mp_drawing.draw_landmarks(image=image,landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=drawing_spec,connection_drawing_spec=drawing_spec)




            if face_results.multi_face_landmarks:
                for landmarks in face_results.multi_face_landmarks:
                    # Extract eye landmarks
                    left_eye_landmarks = [landmarks.landmark[i] for i in range(159, 145, -1)]
                    right_eye_landmarks = [landmarks.landmark[i] for i in range(386, 374, -1)]

                    # Calculate eye aspect ratio (EAR) for left and right eyes
                    left_eye_ear = (cv2.norm(left_eye_landmarks[1].x - left_eye_landmarks[5].x,
                                            left_eye_landmarks[1].y - left_eye_landmarks[5].y) +
                                    cv2.norm(left_eye_landmarks[2].x - left_eye_landmarks[4].x,
                                            left_eye_landmarks[2].y - left_eye_landmarks[4].y)) / \
                                (2 * cv2.norm(left_eye_landmarks[0].x - left_eye_landmarks[3].x,
                                                left_eye_landmarks[0].y - left_eye_landmarks[3].y))

                    right_eye_ear = (cv2.norm(right_eye_landmarks[1].x - right_eye_landmarks[5].x,
                                            right_eye_landmarks[1].y - right_eye_landmarks[5].y) +
                                    cv2.norm(right_eye_landmarks[2].x - right_eye_landmarks[4].x,
                                            right_eye_landmarks[2].y - right_eye_landmarks[4].y)) / \
                                    (2 * cv2.norm(right_eye_landmarks[0].x - right_eye_landmarks[3].x,
                                                right_eye_landmarks[0].y - right_eye_landmarks[3].y))


                    if left_eye_ear < 0.2 or right_eye_ear < 0.2:
                        closed_eye_counter += 1
                    else:
                        closed_eye_counter = 0

                    if closed_eye_counter > 15:  # You can adjust this threshold
                        # cv2.putText(frame, "Sleeping", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        disturbed=True

            # Display the frame
            # cv2.imshow("Frame", frame)
            print(disturbed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

video_thread = threading.Thread(target=video_processing_loop)
video_thread.daemon = True  # Set the thread as a daemon so it exits when the main thread does
video_thread.start()

@app.post("/camera")
async def get_camera_status():
    return {"disturbed": disturbed}

@app.get("/ai")
def read_item(q: Union[str, None] = None):
    #response = ask_ai(q)
    return {"res": "response"}


@app.get("/test")
def read_root(q: Union[str, None] = None):
    return {"res": "its me, why this"+q}



# Load label encoders
le_gender = LabelEncoder()
le_skilllevel = LabelEncoder()

# Load the model and scaler
model_and_scaler_filename = 'completed_course_model_and_scaler.pkl'
model_and_scaler = joblib.load(model_and_scaler_filename)
clf = model_and_scaler['model']
scaler = model_and_scaler['scaler']

# Define Pydantic model for input data
class InputData(BaseModel):
    age: int
    gender: str
    coursecount: int
    timespent: int
    loginstreak: int
    score: int
    codingsolved: int
    skilllevel: str

# Fit label encoders on training data
le_gender.fit(['Male', 'Female'])  # Replace with your actual gender categories
le_skilllevel.fit(['Beginner', 'Intermediate', 'Advanced'])  # Replace with your actual skilllevel categories

@app.post("/predict")
def predict_completion(data:InputData):
    data.dict()
    example_data = {
    "age": int(data.age),
    "gender": str(data.gender),
    "coursecount": int(data.coursecount),
    "timespent": int(data.timespent),
    "loginstreak": int(data.loginstreak),
    "score": int(data.score),
    "codingsolved": int(data.codingsolved),
    "skilllevel": str(data.skilllevel),
}

    example_data["gender"] = le_gender.transform([example_data["gender"]])[0]
    example_data["skilllevel"] = le_skilllevel.transform([example_data["skilllevel"]])[0]
    example_features = ["age", "gender", "coursecount", "timespent", "loginstreak", "score", "codingsolved", "skilllevel"]
    example_df = pd.DataFrame([example_data], columns=example_features)

    # Transform example data using the loaded scaler
    example_scaled = scaler.transform(example_df)

    # Predict using the loaded model
    prediction = clf.predict(example_scaled)

    # Interpret the prediction
    if prediction[0] == 1:
        result = "Completed"
    else:
        result = "Not Completed"

    return {"result": result}
